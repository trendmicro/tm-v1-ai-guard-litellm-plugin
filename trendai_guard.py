import json
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Literal, Optional
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
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    ModelResponse,
    ModelResponseStream,
)

try:
    from litellm._version import version as litellm_version
except Exception:
    litellm_version = "0.0.0"

OPENAI_CHAT_COMPLETION_RESPONSE_V1 = "OpenAIChatCompletionResponseV1"
TMV1_CLIENT_NAME = "litellm"
PLUGIN_VERSION = "v0.1.1"


class GuardRedactedChoiceMessage(BaseModel):
    content: Optional[str] = None


class GuardRedactedChoice(BaseModel):
    message: Optional[GuardRedactedChoiceMessage] = None


class GuardRedactedResponsePayload(BaseModel):
    choices: list[GuardRedactedChoice] = Field(default_factory=list)


class GuardRedactedSimplePayload(BaseModel):
    prompt: Optional[str] = None


class GuardrailResponse(BaseModel):
    action: str = ""
    reasons: list[str] = Field(default_factory=list)
    reason: str = ""
    redactedRequest: Optional[dict] = None


@dataclass
class ScanResult:
    block_reason: Optional[str] = None
    status_code: Optional[int] = None
    # None         → engine returned no redactedRequest field (no PII detected)
    # non-empty str → engine detected PII and returned sanitised text
    redacted_content: Optional[str] = None


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
        **kwargs,
    ):
        super().__init__(**kwargs)

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
        self.stream_batch_size = stream_batch_size
        self.stream_overlap_size = stream_overlap_size

        verbose_proxy_logger.info(
            f"Initialized TrendAI Guard: guardrail_name={self.guardrail_name}, "
            f"api_url={self.api_url}, on_failure={self.on_failure}, "
            f"mask_pii={self.mask_pii}, timeout={self.timeout}"
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

    def _handle_scan_error(
        self,
        description: str,
        block_reason: str,
        status_code: Optional[int] = None,
    ) -> ScanResult:
        """
        Log and return a ScanResult depending on on_failure policy.

        When on_failure == "block", logs at ERROR and returns a blocking ScanResult.
        Otherwise logs at WARNING and returns an empty ScanResult (fail-open).
        """
        if self.on_failure == "block":
            verbose_proxy_logger.error(
                f"TrendAI: {description}; blocking request "
                f"(on_failure={self.on_failure})."
            )
            return ScanResult(
                block_reason=block_reason,
                status_code=status_code,
            )

        verbose_proxy_logger.warning(
            f"TrendAI: {description}; ALLOWING traffic (fail-open mode)."
        )
        return ScanResult()

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
            headers["prefer"] = "redact-pii"
        if request_type:
            headers["TMV1-Request-Type"] = request_type
        return headers

    async def _scan_payload(
        self,
        payload: Any,
        request_type: Optional[str] = None,
    ) -> ScanResult:
        """
        Scans the provided payload using the AI Detection Engine.

        Returns a ScanResult where:
        - block_reason is set when the engine blocked the content
        - redacted_content is a non-empty string when the engine detected PII,
          returned a sanitised version, and action is "allow"
        """
        if payload is None:
            return ScanResult()

        payload_size = len(json.dumps(payload, default=str))

        request_headers = self._build_request_headers(
            request_type=request_type
        )

        verbose_proxy_logger.debug(
            f"TrendAI: Scanning payload (bytes~={payload_size}): {json.dumps(payload, default=str, indent=2)}"
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
                )

            result_json = response.json()
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
                    f"TrendAI: Guard engine blocked content. reason={reason}"
                )
                if redacted_content:
                    return ScanResult(
                        block_reason=(
                            f"Blocked by TrendAI Guard. Security violation: {reason}\n"
                            f"Redacted Payload: {redacted_content}"
                        )
                    )
                return ScanResult(block_reason=f"Blocked by TrendAI Guard. Security violation: {reason}")

            return ScanResult(redacted_content=redacted_content)

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            error_text = e.response.text
            return self._handle_scan_error(
                f"Engine returned status={status_code}; body={error_text}",
                f"Security Guard Error: Engine failure ({status_code})",
                status_code=status_code,
            )
        except Exception as e:
            return self._handle_scan_error(
                f"Unexpected error: {e}",
                f"Security Guard Error: Unable to communicate "
                f"with detection engine ({type(e).__name__})",
                status_code=503,
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
                        return "".join(
                                item.get("text", "")
                                for item in content
                                if isinstance(item, dict) and item.get("type") == "text"
                            ).strip()
                    return None

        texts = inputs.get("texts")
        if isinstance(texts, list):
            prompt = "".join(texts).strip()
            if prompt:
                return prompt

        return None

    @staticmethod
    def _get_chunk_text(
        chunk: ModelResponseStream | ModelResponse,
    ) -> str:
        """Extract text content from a streaming chunk."""
        text_parts = []
        for choice in chunk.choices or []:
            content = (
                choice.delta.content
                if hasattr(choice, "delta")
                else choice.message.content
            )
            if content:
                text_parts.append(content)
        return "".join(text_parts)

    @staticmethod
    def _replace_chunk_content(chunk: Any, content: str) -> bool:
        """Set delta.content on a streaming chunk in-place. Returns True if set."""
        if not hasattr(chunk, "choices") or not chunk.choices:
            return False
        replaced = False
        for choice in chunk.choices:
            if hasattr(choice, "delta") and choice.delta is not None:
                try:
                    choice.delta.content = content
                    replaced = True
                except Exception:
                    pass
        return replaced

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
            response = request_data.get("response")
            if isinstance(response, ModelResponse):
                verbose_proxy_logger.debug(
                    "TrendAI: apply_guardrail response scan triggered"
                )
                response_payload = response.model_dump()
            elif inputs.get("texts"):
                verbose_proxy_logger.debug(
                    "TrendAI: apply_guardrail response scan triggered "
                    "from translated text inputs"
                )
                response_payload = (
                    self._build_chat_completion_payload(
                        "".join(inputs.get("texts", [])),
                        model=(
                            inputs.get("model")
                            if isinstance(inputs.get("model"), str)
                            else "guardrailed-response"
                        ),
                    )
                )
            else:
                verbose_proxy_logger.warning(
                    "TrendAI: apply_guardrail response scan skipped. "
                    "unsupported response shape"
                )
                return inputs

            scan_result = await self._scan_payload(
                response_payload,
                request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
            )

        if scan_result.block_reason:
            raise HTTPException(
                status_code=scan_result.status_code or 400,
                detail={"error": scan_result.block_reason},
            )

        if scan_result.redacted_content:
            texts: list[str] = list(inputs.get("texts") or [])
            if input_type == "request":
                # Locate the last user message content to decide the replacement path.
                last_user_content = None
                structured_messages = inputs.get("structured_messages")
                if isinstance(structured_messages, list):
                    for message in reversed(structured_messages):
                        if message.get("role") == "user":
                            last_user_content = message.get("content")
                            break

                if isinstance(last_user_content, list):
                    # Multi-part array content: LiteLLM appends each text item to
                    # texts in order → they appear as a contiguous suffix.
                    # Replace the last part; zero the rest.
                    parts = [
                        item.get("text", "")
                        for item in last_user_content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    n = len(parts)
                    if n > 0 and len(texts) >= n and texts[-n:] == parts:
                        for j in range(len(texts) - n, len(texts) - 1):
                            texts[j] = ""
                        texts[-1] = scan_result.redacted_content
                        verbose_proxy_logger.debug(
                            "TrendAI: Applied redacted content to request texts "
                            "(multi-part user message, %d parts)", n
                        )
                    else:
                        verbose_proxy_logger.warning(
                            "TrendAI: Request redaction skipped — "
                            "could not match multi-part content in texts"
                        )
                elif "".join(texts).strip() == request_prompt:
                    # No structured_messages (flat list) or single-message string
                    # content where all of texts joins to request_prompt.
                    for i in range(len(texts) - 1, -1, -1):
                        if texts[i].strip():
                            texts[i] = scan_result.redacted_content
                            for j in range(i):
                                texts[j] = ""
                            verbose_proxy_logger.debug(
                                "TrendAI: Applied redacted content to request texts (joined)"
                            )
                            break
                    else:
                        verbose_proxy_logger.warning(
                            "TrendAI: Request redaction skipped — "
                            "could not locate scanned prompt in texts"
                        )
                else:
                    # String content with multiple messages: find by per-element match.
                    for i in range(len(texts) - 1, -1, -1):
                        if texts[i].strip() == request_prompt:
                            texts[i] = scan_result.redacted_content
                            verbose_proxy_logger.debug(
                                "TrendAI: Applied redacted content to request texts[%d]", i
                            )
                            break
                    else:
                        verbose_proxy_logger.warning(
                            "TrendAI: Request redaction skipped — "
                            "could not locate scanned prompt in texts"
                        )
            else:
                # LiteLLM maps texts[i] back to response.choices[i].message.content.
                # Single-choice responses are the common case; for multiple choices the
                # engine returns a joined string so we can only safely replace when
                # there is exactly one text entry.
                if len(texts) == 1:
                    texts[0] = scan_result.redacted_content
                    verbose_proxy_logger.debug(
                        "TrendAI: Applied redacted content to response texts[0]"
                    )
                else:
                    verbose_proxy_logger.warning(
                        "TrendAI: Response redaction skipped — "
                        "cannot distribute redacted text across %d choices",
                        len(texts),
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

        chunk_entries: list[tuple[Any, str]] = []
        buffered_text_chars = 0

        async def _flush_buffer(
            is_final: bool = False,
        ) -> AsyncGenerator[Any, None]:
            nonlocal chunk_entries, buffered_text_chars
            if not chunk_entries:
                return

            full_text = "".join(text for _, text in chunk_entries)
            verbose_proxy_logger.debug(
                f"Flushing streaming buffer. chunk_entries={len(chunk_entries)}, "
                f"buffered_text_chars={buffered_text_chars}, is_final={is_final}, text={full_text}"
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
                scan_result = await self._scan_payload(
                    stream_payload,
                    request_type=OPENAI_CHAT_COMPLETION_RESPONSE_V1,
                )
                if scan_result.block_reason:
                    stream_error = StreamingCallbackError(scan_result.block_reason)
                    if scan_result.status_code is not None:
                        stream_error.status_code = scan_result.status_code
                    raise stream_error

                if scan_result.redacted_content:
                    verbose_proxy_logger.debug(
                        "TrendAI: Applying redacted content to streaming batch"
                    )
                    # Concentrate all redacted text in the last chunk that had
                    # content; zero-out every preceding content chunk.  This
                    # keeps the chunk stream structurally intact (role/finish
                    # chunks pass through unchanged) while replacing PII text.
                    last_content_idx = max(
                        (i for i, (_, t) in enumerate(chunk_entries) if t),
                        default=-1,
                    )
                    new_entries: list[tuple[Any, str]] = []
                    for i, (raw_chunk, orig_text) in enumerate(chunk_entries):
                        if orig_text:
                            new_text = (
                                scan_result.redacted_content
                                if i == last_content_idx
                                else ""
                            )
                            if not self._replace_chunk_content(raw_chunk, new_text):
                                verbose_proxy_logger.warning(
                                    "TrendAI: Failed to replace chunk content in-place; "
                                    "streaming redaction may be incomplete"
                                )
                            new_entries.append((raw_chunk, new_text))
                        else:
                            new_entries.append((raw_chunk, orig_text))
                    chunk_entries = new_entries
                    buffered_text_chars = len(scan_result.redacted_content)

            if is_final:
                for raw_chunk, _ in chunk_entries:
                    yield raw_chunk
                chunk_entries = []
                buffered_text_chars = 0
                return

            # Mid-stream: yield only chunks before the overlap zone
            split_at = buffered_text_chars - self.stream_overlap_size
            cumulative = 0
            split_index = 0
            for i, (_, text) in enumerate(chunk_entries):
                cumulative += len(text)
                if cumulative > split_at:
                    break
                split_index = i + 1

            verbose_proxy_logger.debug(
                f"Overlap split: yielding {split_index}/{len(chunk_entries)} chunks, "
                f"retaining {len(chunk_entries) - split_index} for overlap re-scan"
            )

            for raw_chunk, _ in chunk_entries[:split_index]:
                yield raw_chunk

            # Retain overlap chunks for re-scanning with next batch
            chunk_entries = chunk_entries[split_index:]
            buffered_text_chars = sum(
                len(text) for _, text in chunk_entries
            )

        async for chunk in response:
            if not isinstance(chunk, (ModelResponse, ModelResponseStream)):
                yield chunk
                continue

            chunk_text = self._get_chunk_text(chunk)
            chunk_entries.append((chunk, chunk_text))
            buffered_text_chars += len(chunk_text)

            if buffered_text_chars >= self.stream_batch_size:
                async for flushed_chunk in _flush_buffer(is_final=False):
                    yield flushed_chunk

        if chunk_entries:
            async for flushed_chunk in _flush_buffer(is_final=True):
                yield flushed_chunk
