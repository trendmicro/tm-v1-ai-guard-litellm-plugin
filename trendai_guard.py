import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Literal, Optional, Tuple
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    ModelResponse,
    ModelResponseStream,
)

try:
    from litellm._version import version as litellm_version
except Exception:
    litellm_version = "0.0.0"

OPENAI_CHAT_COMPLETION_RESPONSE_V1 = "OpenAIChatCompletionResponseV1"
RESPONSE_CONTENT_CHUNK_SIZE_BYTES = 49_500
TMV1_CLIENT_NAME = "litellm"
PLUGIN_VERSION = "v0.1.2"


class GuardRedactedChoiceMessage(BaseModel):
    content: Optional[str] = None


class GuardRedactedChoice(BaseModel):
    message: Optional[GuardRedactedChoiceMessage] = None


class GuardRedactedResponsePayload(BaseModel):
    choices: list[GuardRedactedChoice] = Field(default_factory=list)


class GuardRedactedSimplePayload(BaseModel):
    prompt: Optional[str] = None


class GuardSensitiveRule(BaseModel):
    id: str = ""


class GuardSensitiveInformation(BaseModel):
    hasPolicyViolation: bool = False
    rules: list[GuardSensitiveRule] = Field(default_factory=list)


class GuardrailResponse(BaseModel):
    action: str = ""
    reasons: list[str] = Field(default_factory=list)
    reason: str = ""
    redactedRequest: Optional[dict] = None
    sensitiveInformation: Optional[GuardSensitiveInformation] = None


@dataclass
class ScanResult:
    block_reason: Optional[str] = None
    status_code: Optional[int] = None
    # None         → engine returned no redactedRequest field (no PII detected)
    # non-empty str → engine detected PII and returned sanitised text
    redacted_content: Optional[str] = None
    engine_errored: bool = False
    # {entity_type: count} of sensitive-data entities the engine masked; set
    # only on the allow + redaction path (None on block, engine error, or when
    # no redaction was applied).
    masked_entity_count: Optional[Dict[str, int]] = None


@dataclass
class _ScanAggregate:
    block_reason: Optional[str] = None
    status_code: Optional[int] = None
    engine_errored: bool = False
    redacted: bool = False
    masked_entity_count: Dict[str, int] = field(default_factory=dict)

    def add(self, result: ScanResult) -> None:
        self.engine_errored = self.engine_errored or result.engine_errored
        self.redacted = self.redacted or result.redacted_content is not None
        if result.block_reason and self.block_reason is None:
            self.block_reason = result.block_reason
            self.status_code = result.status_code
        for entity_type, count in (result.masked_entity_count or {}).items():
            # Overlapping windows may report the same entity more than once.
            self.masked_entity_count[entity_type] = max(
                self.masked_entity_count.get(entity_type, 0), count
            )

    @property
    def terminal(self) -> bool:
        return self.engine_errored or self.block_reason is not None

    def as_scan_result(self) -> ScanResult:
        return ScanResult(
            block_reason=self.block_reason,
            status_code=self.status_code,
            engine_errored=self.engine_errored,
            masked_entity_count=self.masked_entity_count or None,
        )


@dataclass
class _ResponseWindowScan:
    original: str
    start: int
    result: ScanResult


@dataclass
class _ChunkedResponseScan:
    windows: list[_ResponseWindowScan] = field(default_factory=list)
    aggregate: _ScanAggregate = field(default_factory=_ScanAggregate)


@dataclass
class _StreamChunkEntry:
    chunk: Any
    original: str
    output: str


class TrendAIGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        app_name: Optional[str] = None,
        fallback_on_error: Literal["block", "allow"] = "block",
        mask_pii: bool = True,
        timeout: float = 5.0,
        stream_batch_size: int = 2048,
        stream_overlap_size: int = 256,
        response_content_chunk_size_bytes: int = RESPONSE_CONTENT_CHUNK_SIZE_BYTES,
        logging_only_scan: Literal["request", "response", "both"] = "both",
        **kwargs,
    ):
        super().__init__(
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.logging_only,
            ],
            **kwargs,
        )

        self.api_url = (
            api_base
            or os.getenv("TRENDAI_AI_GUARD_BASE_URL")
        )
        if not self.api_url:
            raise ValueError(
                "TrendAI Guard Plugin requires an API base URL. "
                "Set api_base parameter in litellm config or TRENDAI_AI_GUARD_BASE_URL environment variable."
            )

        self.tmv1_app_name = (
            app_name or os.getenv("TMV1_APPLICATION_NAME", "litellm")
        )
        self.tmv1_api_key = api_key or os.getenv("TMV1_API_KEY")
        if not self.tmv1_api_key:
            raise ValueError(
                "TrendAI Guard Plugin requires an API key. "
                "Set api_key parameter in litellm config or TMV1_API_KEY environment variable."
            )
        self.on_failure = fallback_on_error
        self.mask_pii = mask_pii
        self.timeout = timeout
        self.stream_batch_size = max(1, stream_batch_size)
        self.stream_overlap_size = min(
            max(0, stream_overlap_size), self.stream_batch_size - 1
        )
        self.response_content_chunk_size_bytes = max(
            1, response_content_chunk_size_bytes
        )

        if logging_only_scan not in ("request", "response", "both"):
            raise ValueError(
                "TrendAI Guard Plugin logging_only_scan must be one of "
                f"'request', 'response', or 'both'; got {logging_only_scan!r}."
            )
        self.logging_only_scan = logging_only_scan

        verbose_proxy_logger.info(
            f"Initialized TrendAI Guard: guardrail_name={self.guardrail_name}, "
            f"api_url={self.api_url}, on_failure={self.on_failure}, "
            f"mask_pii={self.mask_pii}, timeout={self.timeout}, "
            f"logging_only_scan={self.logging_only_scan}"
        )

    @staticmethod
    def _parse_model(
        model_cls: type[BaseModel], payload: Any
    ) -> Optional[BaseModel]:
        try:
            model_validate = getattr(model_cls, "model_validate", None)
            if callable(model_validate):
                return model_validate(payload)
        except Exception:
            return None
        return None

    def _extract_redacted_content(
        self, request_type: Optional[str], redacted_payload: Any
    ) -> Optional[str]:
        """
        Extract sanitised text from the engine's redactedRequest field.
        Returns None when parsing fails; otherwise a non-empty sanitised string.
        """
        if request_type == OPENAI_CHAT_COMPLETION_RESPONSE_V1:
            parsed = self._parse_model(
                GuardRedactedResponsePayload, redacted_payload
            )
            if parsed is None:
                return None
            return " ".join(
                choice.message.content
                for choice in parsed.choices
                if choice.message is not None and choice.message.content
            ).strip() or None

        parsed = self._parse_model(GuardRedactedSimplePayload, redacted_payload)
        if parsed is None:
            return None
        return parsed.prompt or None

    def _emit_guardrail_log(
        self,
        *,
        request_data: dict,
        guardrail_json_response: dict,
        guardrail_status: GuardrailStatus,
        start_time: Optional[float],
        event_type: Optional[GuardrailEventHooks],
        masked_entity_count: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Stamp end_time/duration and write one standard-logging guardrail entry.

        Shared by the scan path (_scan_payload, _handle_scan_error) and the
        logging_only hook so every entry is shaped consistently.
        """
        end_time = datetime.now().timestamp()
        duration = (
            (end_time - start_time) if start_time is not None else None
        )
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_json_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            event_type=event_type,
            masked_entity_count=masked_entity_count,
        )

    def _handle_scan_error(
        self,
        description: str,
        block_reason: str,
        status_code: Optional[int] = None,
        request_data: Optional[dict] = None,
        start_time: Optional[float] = None,
        event_type: Optional[GuardrailEventHooks] = None,
    ) -> ScanResult:
        """
        Log and return a ScanResult depending on on_failure policy.

        When on_failure == "block", logs at ERROR and returns a blocking ScanResult.
        Otherwise logs at WARNING and returns an empty ScanResult (fail-open).

        When request_data is provided, emits a standard-logging entry so downstream
        observability systems (Langfuse, Datadog, OTEL) can track guardrail failures.
        """
        if request_data is not None:
            # on_failure governs whether the request is blocked, not how the
            # outage is classified: always failed_to_respond, never intervened
            # (which would masquerade an outage as a security detection).
            self._emit_guardrail_log(
                request_data=request_data,
                guardrail_json_response={
                    "description": description,
                    "block_reason": block_reason,
                    "status_code": status_code,
                },
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time,
                event_type=event_type,
            )

        if self.on_failure == "block":
            verbose_proxy_logger.error(
                f"TrendAI: {description}; blocking request "
                f"(on_failure={self.on_failure})."
            )
            return ScanResult(
                block_reason=block_reason,
                status_code=status_code,
                engine_errored=True,
            )

        verbose_proxy_logger.warning(
            f"TrendAI: {description}; ALLOWING traffic (fail-open mode)."
        )
        return ScanResult(engine_errored=True)

    def _build_request_headers(
        self,
        request_type: Optional[str] = None,
    ) -> dict[str, str]:
        headers = {
            "TMV1-Application-Name": self.tmv1_app_name,
            "Authorization": f"Bearer {self.tmv1_api_key}",
            "Content-Type": "application/json",
            "TMV1-Client-Name": TMV1_CLIENT_NAME,
            "TMV1-Client-Version": litellm_version,
            "TMV1-Plugin-Version": PLUGIN_VERSION,
        }
        if self.mask_pii:
            headers["prefer"] = "redact-pii,return=representation"
        if request_type:
            headers["TMV1-Request-Type"] = request_type
        return headers

    @staticmethod
    def _extract_masked_entity_count(
        sensitive_information: Optional[GuardSensitiveInformation],
    ) -> Optional[Dict[str, int]]:
        """
        Build a {entity_type: count} map from the engine's sensitiveInformation
        rules for standard-logging observability (OTEL/Langfuse/Datadog).

        The engine deduplicates hits by entity type, so counts are typically 1
        per type; we still aggregate defensively. Returns None when there are no
        rules so no masked_entity_count is attached.
        """
        if sensitive_information is None:
            return None
        counts: Dict[str, int] = {}
        for rule in sensitive_information.rules:
            entity_type = (rule.id or "").strip()
            if entity_type:
                counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts or None

    async def _scan_payload(
        self,
        payload: Any,
        request_type: Optional[str] = None,
        request_data: Optional[dict] = None,
        event_type: Optional[GuardrailEventHooks] = None,
        log_success: bool = True,
    ) -> ScanResult:
        """
        Scans the provided payload using the AI Detection Engine.

        When ``log_success`` is False, a successful (allow) scan is not written
        to standard logging. Blocks and engine errors are always logged. This
        lets the streaming hook scan each batch for early blocking while
        emitting only a single "success" entry (on the final flush), matching
        the one-entry-per-response convention of other LiteLLM guardrails.

        Returns a ScanResult where:
        - block_reason is set when the engine blocked the content
        - redacted_content is a non-empty string when the engine detected PII,
          returned a sanitised version, and action is "allow"
        """
        start_time = datetime.now().timestamp()

        if payload is None:
            return ScanResult()

        payload_size = len(json.dumps(payload, default=str))

        request_headers = self._build_request_headers(
            request_type=request_type
        )

        if verbose_proxy_logger.isEnabledFor(logging.DEBUG):
            verbose_proxy_logger.debug(
                "TrendAI: Scanning payload (bytes~=%d): %s",
                payload_size,
                json.dumps(payload, default=str, indent=2),
            )

        try:
            url = self.api_url
            if not url.endswith("/applyGuardrails"):
                parsed = urlparse(url)
                if (
                    parsed.hostname
                    and parsed.hostname.endswith(".trendmicro.com")
                    and "/aiSecurity" not in parsed.path
                ):
                    url = f"{url.rstrip('/')}/v3.0/aiSecurity"
                url = f"{url.rstrip('/')}/applyGuardrails"

            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.GuardrailCallback
            )

            response = await async_client.post(
                url,
                json=payload,
                headers=request_headers,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                error_text = response.text
                status_code = response.status_code
                return self._handle_scan_error(
                    f"Engine returned status={status_code}; body={error_text}",
                    f"Security Guard Error: Engine failure ({status_code})",
                    status_code=status_code,
                    request_data=request_data,
                    start_time=start_time,
                    event_type=event_type,
                )

            result_json = response.json()
            verbose_proxy_logger.debug(
                "TrendAI: Guard engine response: %s", result_json
            )
            parsed_result = self._parse_model(GuardrailResponse, result_json)
            if parsed_result is None:
                parsed_result = GuardrailResponse()

            redacted_content: Optional[str] = None
            if parsed_result.redactedRequest is not None:
                verbose_proxy_logger.debug(
                    "TrendAI: Guard engine returned redacted payload"
                )
                redacted_content = self._extract_redacted_content(
                    request_type, parsed_result.redactedRequest
                )

            action = parsed_result.action
            if action.lower() == "block":
                reasons = parsed_result.reasons
                reason = (
                    ", ".join(reasons) if reasons else parsed_result.reason
                )
                verbose_proxy_logger.debug(
                    "TrendAI: Guard engine blocked content. reason=%s", reason
                )
                block_reason_text = (
                    f"Blocked by TrendAI Guard. Security violation: {reason}\n"
                    f"Redacted Payload: {redacted_content}"
                    if redacted_content
                    else f"Blocked by TrendAI Guard. Security violation: {reason}"
                )
                # No masked_entity_count on block: the request is rejected, so no
                # masked content is delivered. Detected entities are conveyed via
                # reasons and guardrail_status="guardrail_intervened".
                if request_data is not None:
                    self._emit_guardrail_log(
                        request_data=request_data,
                        guardrail_json_response={
                            "action": action,
                            "reason": reason,
                            "reasons": reasons,
                        },
                        guardrail_status="guardrail_intervened",
                        start_time=start_time,
                        event_type=event_type,
                    )
                if redacted_content:
                    return ScanResult(block_reason=block_reason_text)
                return ScanResult(
                    block_reason=f"Blocked by TrendAI Guard. Security violation: {reason}"
                )

            # Allow path: only report entities actually masked, i.e. the engine
            # returned sanitised content that flows onward. Detections without
            # redaction (e.g. mask_pii disabled) are not "masked".
            masked_entity_count = (
                self._extract_masked_entity_count(
                    parsed_result.sensitiveInformation
                )
                if redacted_content
                else None
            )

            if request_data is not None and log_success:
                self._emit_guardrail_log(
                    request_data=request_data,
                    guardrail_json_response={
                        "action": action,
                        "redacted": redacted_content is not None,
                    },
                    guardrail_status="success",
                    start_time=start_time,
                    event_type=event_type,
                    masked_entity_count=masked_entity_count,
                )

            return ScanResult(
                redacted_content=redacted_content,
                masked_entity_count=masked_entity_count,
            )

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_text = e.response.text
            return self._handle_scan_error(
                f"Engine returned status={status_code}; body={error_text}",
                f"Security Guard Error: Engine failure ({status_code})",
                status_code=status_code,
                request_data=request_data,
                start_time=start_time,
                event_type=event_type,
            )
        except Exception as e:
            return self._handle_scan_error(
                f"Unexpected error: {e}",
                f"Security Guard Error: Unable to communicate "
                f"with detection engine ({type(e).__name__})",
                status_code=503,
                request_data=request_data,
                start_time=start_time,
                event_type=event_type,
            )

    def _build_chat_completion_payload(
        self, content: str, model: str = "streaming-response"
    ) -> dict:
        return {
            "id": "chatcmpl-stream",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    @staticmethod
    def _chunk_text_by_utf8_bytes(
        content: str,
        overlap_size: int = 0,
        chunk_size_bytes: Optional[int] = None,
    ) -> list[str]:
        """Split content into byte-limited windows with character overlap."""
        encoded_content = content.encode("utf-8")
        overlap_size = max(overlap_size, 0)
        chunk_size_bytes = max(
            RESPONSE_CONTENT_CHUNK_SIZE_BYTES
            if chunk_size_bytes is None
            else chunk_size_bytes,
            1,
        )
        chunks: list[str] = []
        offset = 0

        while offset < len(encoded_content):
            candidate = encoded_content[offset:offset + chunk_size_bytes]
            # The input is valid UTF-8, so only a partial trailing character
            # can be ignored here.
            chunk = candidate.decode("utf-8", errors="ignore")
            chunks.append(chunk)

            consumed_bytes = len(chunk.encode("utf-8"))
            offset += consumed_bytes
            if offset == len(encoded_content):
                break

            overlap_chars = min(overlap_size, len(chunk) - 1)
            if overlap_chars:
                offset -= len(chunk[-overlap_chars:].encode("utf-8"))

        return chunks

    def _get_request_prompt_from_inputs(
        self, inputs: GenericGuardrailAPIInputs
    ) -> Optional[str]:
        structured_messages = inputs.get("structured_messages")
        if isinstance(structured_messages, list):
            for message in reversed(structured_messages):
                if message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
                    elif isinstance(content, list):
                        return "".join(self._extract_text_parts(content)).strip()
                    return None

        texts = inputs.get("texts")
        if isinstance(texts, list):
            prompt = "".join(texts).strip()
            if prompt:
                return prompt

        return None

    @staticmethod
    def _extract_text_parts(content: Any) -> list[str]:
        """Return text strings from a multi-part content array."""
        if not isinstance(content, list):
            return []
        return [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]

    @staticmethod
    def _is_primary_choice(choice: Any, position: int) -> bool:
        """Return whether a choice is the single supported response choice."""
        index = getattr(choice, "index", None)
        return index == 0 if isinstance(index, int) else position == 0

    def _has_unsupported_choices(self, chunk: Any) -> bool:
        """Return True when a response contains a choice other than index 0."""
        choices = getattr(chunk, "choices", None) or []
        return len(choices) > 1 or any(
            not self._is_primary_choice(choice, position)
            for position, choice in enumerate(choices)
        )

    @staticmethod
    def _get_chunk_text(chunk: Any) -> str:
        """Extract text content from a streaming chunk."""
        if isinstance(chunk, bytes):
            try:
                decoded = chunk.decode("utf-8", errors="replace")
                parts = []
                for line in decoded.split("\n"):
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            if (
                                data.get("type") == "content_block_delta"
                                and (data.get("delta") or {}).get("type") == "text_delta"
                            ):
                                parts.append((data.get("delta") or {}).get("text") or "")
                        except (json.JSONDecodeError, ValueError):
                            pass
                return "".join(parts)
            except Exception:
                pass
            return ""
        for position, choice in enumerate(chunk.choices or []):
            if not TrendAIGuardrail._is_primary_choice(choice, position):
                continue
            content = (
                choice.delta.content
                if hasattr(choice, "delta")
                else choice.message.content
            )
            return content or ""
        return ""

    def _replace_chunk_content(self, chunk: Any, content: str) -> bool:
        """Set delta.content on the supported streaming choice in-place."""
        if not hasattr(chunk, "choices") or not chunk.choices:
            return False
        for position, choice in enumerate(chunk.choices):
            if not self._is_primary_choice(choice, position):
                continue
            if hasattr(choice, "delta") and choice.delta is not None:
                try:
                    choice.delta.content = content
                    return True
                except Exception:
                    pass
        return False

    @staticmethod
    def _merge_redaction(original: str, output: str, redacted: str) -> str:
        """Apply new masks without undoing masks from an overlapping scan.

        If the engine returns extra trailing content, trim it to the original
        length. Other length mismatches are unexpected and cannot be merged
        positionally, so retain the prior output.
        """
        if len(redacted) > len(original):
            verbose_proxy_logger.warning(
                "TrendAI: Response redaction is longer than the original; "
                "trimming redacted content (original=%d, output=%d, redacted=%d)",
                len(original),
                len(output),
                len(redacted),
            )
            redacted = redacted[: len(original)]
        if len(redacted) < len(original) or len(output) != len(original):
            verbose_proxy_logger.warning(
                "TrendAI: Unexpected response redaction length conflict; "
                "preserving prior output "
                "(original=%d, output=%d, redacted=%d)",
                len(original),
                len(output),
                len(redacted),
            )
            return output
        return "".join(
            redacted_char if redacted_char != original_char else output_char
            for original_char, output_char, redacted_char in zip(
                original, output, redacted
            )
        )

    async def _scan_response_windows(
        self, content: str, model: str
    ) -> _ChunkedResponseScan:
        """Scan UTF-8-limited overlapping response windows until terminal."""
        chunked_scan = _ChunkedResponseScan()
        chunk_start = 0
        leading_overlap = 0
        chunks = self._chunk_text_by_utf8_bytes(
            content,
            self.stream_overlap_size,
            self.response_content_chunk_size_bytes,
        )

        for index, original in enumerate(chunks):
            result = await self._scan_payload(
                self._build_chat_completion_payload(original, model=model),
                request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
            )
            chunked_scan.windows.append(
                _ResponseWindowScan(
                    original=original,
                    start=chunk_start,
                    result=result,
                )
            )
            chunked_scan.aggregate.add(result)
            if chunked_scan.aggregate.terminal:
                break

            if index < len(chunks) - 1:
                leading_overlap = min(
                    self.stream_overlap_size, max(len(original) - 1, 0)
                )
                chunk_start += len(original) - leading_overlap

        return chunked_scan

    @staticmethod
    def _replace_bytes_content(chunk: bytes, content: str) -> Optional[bytes]:
        """Replace text_delta text in an Anthropic SSE bytes chunk.

        Returns new bytes with the text replaced, or None if the chunk
        does not contain a text_delta or replacement fails.
        """
        try:
            decoded = chunk.decode("utf-8", errors="replace")
            lines = decoded.split("\n")
            text_delta_indices = []
            for idx, line in enumerate(lines):
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if (
                            data.get("type") == "content_block_delta"
                            and (data.get("delta") or {}).get("type") == "text_delta"
                        ):
                            text_delta_indices.append((idx, data))
                    except (json.JSONDecodeError, ValueError):
                        pass
            if not text_delta_indices:
                return None
            for pos, (idx, data) in enumerate(text_delta_indices):
                data["delta"]["text"] = content if pos == len(text_delta_indices) - 1 else ""
                lines[idx] = f"data: {json.dumps(data)}"
            return "\n".join(lines).encode("utf-8")
        except Exception:
            return None

    def _redact_texts_for_request(
        self,
        texts: list[str],
        redacted_content: str,
        request_prompt: Optional[str],
        structured_messages: Optional[list[dict]] = None,
    ) -> bool:
        """
        Find and replace the scanned prompt in ``texts`` with ``redacted_content``
        in-place. Returns True when the replacement succeeded.
        """
        if not request_prompt:
            return False

        last_user_content = None
        if isinstance(structured_messages, list):
            for message in reversed(structured_messages):
                if message.get("role") == "user":
                    last_user_content = message.get("content")
                    break

        if isinstance(last_user_content, list):
            parts = self._extract_text_parts(last_user_content)
            n = len(parts)
            if n > 0 and len(texts) >= n and texts[-n:] == parts:
                for j in range(len(texts) - n, len(texts) - 1):
                    texts[j] = ""
                texts[-1] = redacted_content
                return True

        if "".join(texts).strip() == request_prompt:
            for i in range(len(texts) - 1, -1, -1):
                if texts[i].strip():
                    texts[i] = redacted_content
                    for j in range(i):
                        texts[j] = ""
                    return True

        for i in range(len(texts) - 1, -1, -1):
            if texts[i].strip() == request_prompt:
                texts[i] = redacted_content
                return True

        return False

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Unified guardrail entrypoint for request scanning and non-streaming
        response scanning.

        When the engine returns a redacted payload with action "allow", the
        sanitised text is written back into inputs["texts"] so LiteLLM's
        translation layer propagates it to the actual LLM call (pre-call) or
        the client response (post-call).
        """

        request_prompt: Optional[str] = None

        if input_type == "request":
            request_prompt = self._get_request_prompt_from_inputs(inputs)
            if request_prompt:
                verbose_proxy_logger.debug(
                    "TrendAI: apply_guardrail request scan triggered "
                    "from normalized inputs"
                )
                scan_result = await self._scan_payload(
                    {"prompt": request_prompt}
                )
            else:
                verbose_proxy_logger.warning(
                    "TrendAI: apply_guardrail request scan skipped. "
                    "No prompt extracted from inputs"
                )
                return inputs
        else:
            texts: list[str] = list(inputs.get("texts") or [])
            if texts:
                verbose_proxy_logger.debug(
                    "TrendAI: apply_guardrail response scan triggered "
                    "from translated text inputs"
                )
                if len(texts) > 1:
                    message = (
                        "TrendAI: Multiple response choices are not supported"
                    )
                    if self.on_failure == "block":
                        raise HTTPException(
                            status_code=400,
                            detail={"error": message},
                        )
                    verbose_proxy_logger.warning(
                        "%s; scanning only the first choice "
                        "(fallback_on_error=%s).",
                        message,
                        self.on_failure,
                    )
                response_content = texts[0]
            else:
                verbose_proxy_logger.warning(
                    "TrendAI: apply_guardrail response scan skipped. "
                    "unsupported response shape"
                )
                return inputs

            model = (
                inputs.get("model")
                if isinstance(inputs.get("model"), str)
                else "guardrailed-response"
            )
            chunked_scan = await self._scan_response_windows(
                response_content, model
            )
            aggregate = chunked_scan.aggregate
            if aggregate.block_reason:
                raise HTTPException(
                    status_code=aggregate.status_code or 400,
                    detail={"error": aggregate.block_reason},
                )

            merged_content = response_content
            for window in chunked_scan.windows:
                redacted_chunk = window.result.redacted_content
                if redacted_chunk is None:
                    continue
                current = merged_content[
                    window.start:window.start + len(window.original)
                ]
                merged = self._merge_redaction(
                    window.original, current, redacted_chunk
                )
                merged_content = (
                    merged_content[:window.start]
                    + merged
                    + merged_content[window.start + len(window.original):]
                )

            inputs["texts"] = [merged_content, *texts[1:]]
            return inputs

        if scan_result.block_reason:
            raise HTTPException(
                status_code=scan_result.status_code or 400,
                detail={"error": scan_result.block_reason},
            )

        if scan_result.redacted_content:
            texts: list[str] = list(inputs.get("texts") or [])
            if not self._redact_texts_for_request(
                texts,
                scan_result.redacted_content,
                request_prompt,
                inputs.get("structured_messages"),
            ):
                verbose_proxy_logger.warning(
                    "TrendAI: Request redaction skipped — "
                    "could not locate scanned prompt in texts"
                )
            inputs["texts"] = texts

        return inputs

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        """
        Hook called for streaming responses.
        Performs batch scans before yielding buffered chunks.
        Uses a sliding-window overlap so that the last overlap_size
        characters of each batch are re-scanned with the next batch,
        covering security violations that span batch boundaries.
        """
        from litellm.proxy.proxy_server import StreamingCallbackError

        # Retained entries keep immutable scanner input separate from output
        # containing masks accumulated across overlapping scans.
        chunk_entries: list[_StreamChunkEntry] = []
        buffered_text_chars = 0
        buffer_has_unscanned_text = False
        stream_scanned = False
        unsupported_choices_seen = False
        aggregate = _ScanAggregate()
        stream_start = datetime.now().timestamp()

        async def _flush_buffer(
            is_final: bool = False,
        ) -> AsyncGenerator[Any, None]:
            nonlocal chunk_entries, buffered_text_chars, buffer_has_unscanned_text
            nonlocal stream_scanned
            if not chunk_entries:
                return

            full_text = (
                "".join(entry.original for entry in chunk_entries)
                if buffer_has_unscanned_text
                else ""
            )
            verbose_proxy_logger.debug(
                "Flushing streaming buffer. chunk_entries=%d, "
                "buffered_text_chars=%d, is_final=%s, text=%s",
                len(chunk_entries),
                buffered_text_chars,
                is_final,
                full_text,
            )
            if full_text:
                stream_model = (
                    request_data.get("model", "streaming-response")
                    if isinstance(request_data, dict)
                    else "streaming-response"
                )
                stream_payload = self._build_chat_completion_payload(
                    full_text, stream_model
                )
                # Log success only on the final flush, else each batch would
                # emit a duplicate "success" record for one streamed response.
                # Blocks and engine errors are always logged.
                scan_result = await self._scan_payload(
                    stream_payload,
                    request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
                    request_data=request_data,
                    event_type=GuardrailEventHooks.post_call,
                    log_success=False,
                )
                stream_scanned = True
                aggregate.add(scan_result)
                if scan_result.block_reason:
                    stream_error = StreamingCallbackError(scan_result.block_reason)
                    if scan_result.status_code is not None:
                        stream_error.status_code = scan_result.status_code
                    raise stream_error

                if scan_result.redacted_content:
                    verbose_proxy_logger.debug(
                        "TrendAI: Applying redacted content to streaming batch"
                    )
                    merged_output = self._merge_redaction(
                        full_text,
                        "".join(entry.output for entry in chunk_entries),
                        scan_result.redacted_content,
                    )
                    redacted_offset = 0
                    for i, entry in enumerate(chunk_entries):
                        text_length = len(entry.original)
                        if entry.original:
                            if isinstance(entry.chunk, bytes):
                                bytes_text = merged_output[
                                    redacted_offset:redacted_offset + text_length
                                ]
                                new_chunk = self._replace_bytes_content(entry.chunk, bytes_text)
                                if new_chunk is None:
                                    verbose_proxy_logger.error(
                                        "TrendAI: Failed to replace bytes chunk content; "
                                        "blocking to prevent PII leak"
                                    )
                                    raise StreamingCallbackError(
                                        "Redaction of streaming content failed"
                                    )
                                entry.chunk = new_chunk
                                entry.output = bytes_text
                            else:
                                entry.output = merged_output[
                                    redacted_offset:redacted_offset + text_length
                                ]
                                if not self._replace_chunk_content(entry.chunk, entry.output):
                                    verbose_proxy_logger.warning(
                                        "TrendAI: Failed to replace chunk content in-place; "
                                        "streaming redaction may be incomplete"
                                    )
                        redacted_offset += text_length
                buffer_has_unscanned_text = False

            if is_final:
                for entry in chunk_entries:
                    yield entry.chunk
                chunk_entries = []
                buffered_text_chars = 0
                return

            split_at = buffered_text_chars - self.stream_overlap_size
            cumulative = 0
            split_index = 0
            for index, entry in enumerate(chunk_entries):
                cumulative += len(entry.original)
                if cumulative > split_at:
                    break
                split_index = index + 1

            verbose_proxy_logger.debug(
                "Overlap split: yielding %d/%d chunks, "
                "retaining %d for overlap re-scan",
                split_index,
                len(chunk_entries),
                len(chunk_entries) - split_index,
            )

            yielded_chars = 0
            for entry in chunk_entries[:split_index]:
                yielded_chars += len(entry.original)
                yield entry.chunk

            chunk_entries = chunk_entries[split_index:]
            buffered_text_chars -= yielded_chars

        async for chunk in response:
            if not isinstance(chunk, (ModelResponse, ModelResponseStream, bytes)):
                yield chunk
                continue

            if self._has_unsupported_choices(chunk):
                message = "TrendAI: Multiple response choices are not supported"
                if self.on_failure == "block":
                    stream_error = StreamingCallbackError(message)
                    stream_error.status_code = 400
                    raise stream_error
                if not unsupported_choices_seen:
                    verbose_proxy_logger.warning(
                        "%s; scanning only choice index 0 "
                        "(fallback_on_error=%s).",
                        message,
                        self.on_failure,
                    )
                    unsupported_choices_seen = True

            chunk_text = self._get_chunk_text(chunk)
            chunk_entries.append(
                _StreamChunkEntry(chunk, chunk_text, chunk_text)
            )
            buffered_text_chars += len(chunk_text)
            buffer_has_unscanned_text = buffer_has_unscanned_text or bool(chunk_text)

            if buffered_text_chars >= self.stream_batch_size:
                async for flushed_chunk in _flush_buffer(is_final=False):
                    yield flushed_chunk

        async for flushed_chunk in _flush_buffer(is_final=True):
            yield flushed_chunk

        if stream_scanned:
            # Per-batch successes are suppressed during scanning; emit one
            # aggregate success record for the complete stream.
            self._emit_guardrail_log(
                request_data=request_data,
                guardrail_json_response={
                    "action": "allow",
                    "redacted": aggregate.redacted,
                },
                guardrail_status="success",
                start_time=stream_start,
                event_type=GuardrailEventHooks.post_call,
                masked_entity_count=aggregate.masked_entity_count or None,
            )

    def _log_logging_only_scan(
        self, scan_result: ScanResult, request_data: dict, start_time: float
    ) -> None:
        """Map a ScanResult to a single logging_only standard-logging entry.

        engine_errored is checked first: a fail-closed engine outage also sets
        block_reason, but it is an outage, not a detection — so it is recorded
        as guardrail_failed_to_respond and the engine-error text is not
        surfaced as a block_reason.
        """
        status: GuardrailStatus = "success"
        body: dict[str, Any] = {}
        if scan_result.engine_errored:
            status = "guardrail_failed_to_respond"
        elif scan_result.block_reason:
            status = "guardrail_intervened"
            body["block_reason"] = scan_result.block_reason
        # masked_entity_count is populated by _scan_payload only on the allow +
        # redaction path, so it is None for blocks and engine errors.
        self._emit_guardrail_log(
            request_data=request_data,
            guardrail_json_response=body,
            guardrail_status=status,
            start_time=start_time,
            event_type=GuardrailEventHooks.logging_only,
            masked_entity_count=scan_result.masked_entity_count,
        )

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Scan request/response for observability without blocking.
        """
        scan_request = self.logging_only_scan in ("request", "both")
        scan_response = self.logging_only_scan in ("response", "both")

        try:
            # Scan request messages
            messages = kwargs.get("messages")
            if scan_request and isinstance(messages, list):
                inputs: GenericGuardrailAPIInputs = {
                    "structured_messages": messages,
                    "texts": [
                        m.get("content")
                        for m in messages
                        if isinstance(m.get("content"), str)
                    ],
                }
                prompt = self._get_request_prompt_from_inputs(inputs)
                if prompt:
                    request_start = datetime.now().timestamp()
                    request_scan = await self._scan_payload({"prompt": prompt})
                    self._log_logging_only_scan(
                        request_scan, kwargs, request_start
                    )

            # Scan response. Use the same UTF-8 byte limit and overlap as the
            # enforcing non-streaming post-call path, but never mutate result.
            if scan_response and isinstance(result, ModelResponse) and result.choices:
                if self._has_unsupported_choices(result):
                    verbose_proxy_logger.warning(
                        "TrendAI: Multiple response choices are not supported; "
                        "logging_only scans only choice index 0."
                    )
                response_content = ""
                for position, choice in enumerate(result.choices):
                    if not self._is_primary_choice(choice, position):
                        continue
                    if (
                        getattr(choice, "message", None) is not None
                        and isinstance(choice.message.content, str)
                    ):
                        response_content = choice.message.content
                    break
                if response_content:
                    response_start = datetime.now().timestamp()
                    result_model = getattr(result, "model", None)
                    model = (
                        result_model
                        if isinstance(result_model, str) and result_model
                        else kwargs.get("model", "guardrailed-response")
                    )
                    chunked_scan = await self._scan_response_windows(
                        response_content, model
                    )
                    self._log_logging_only_scan(
                        chunked_scan.aggregate.as_scan_result(),
                        kwargs,
                        response_start,
                    )

        except Exception as e:
            verbose_proxy_logger.warning(
                "TrendAI logging_only scan error: %s", str(e)
            )
            self._emit_guardrail_log(
                request_data=kwargs,
                guardrail_json_response={"error": str(e)},
                guardrail_status="guardrail_failed_to_respond",
                start_time=datetime.now().timestamp(),
                event_type=GuardrailEventHooks.logging_only,
            )

        return kwargs, result

    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """No-op: all scanning is handled by async_logging_hook.
        """
        return kwargs, result
