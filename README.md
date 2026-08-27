# TrendAI LiteLLM Guardrail

Integrate the TrendAI Vision One™ AI Guard API with [LiteLLM](https://docs.litellm.ai/docs/) to enforce content guardrails on large language model requests and responses.

## Prerequisites

### Obtain an API Key

1. In the Vision One console, navigate to **Workflow and Automation** > **Third-Party Integrations** > **LiteLLM**.
2. Click **Generate Token** to create an API key.

**Note:** The account used to generate the API key must be in the same V1 region where your LiteLLM deployment resides.

### Obtain the AI Guard Base URL

The procedure for obtaining the base URL depends on your AI Guard deployment model.

**TrendAI Hosted**

On the LiteLLM third-party integrations page in TrendAI Vision One™, copy the **Endpoint URL**.

**Self-Hosted AWS**

1. In the AWS management console, open the AI Guard CloudFormation stack and click the **Outputs** tab.
2. Copy `GuardAPIEndpoint`.

## Configure the LiteLLM Guardrail

### Required Parameters

| Parameter | Description | Use |
|-----------|-------------|-----|
| `guardrail` | Specifies the guardrail module. | Set to `trendai_guard.TrendAIGuardrail`. |
| `api_base` | Base URL for the AI Guard API. | Examples: `https://api.xdr.trendmicro.com/v3.0/aiSecurity`, `https://api.eu.xdr.trendmicro.com/v3.0/aiSecurity` |
| `api_key` | API key for authenticating with the AI Guard API. | Set this as an environment variable (e.g., `os.environ/TMV1_API_KEY`) to avoid storing credentials in configuration files. |
| `mode` | Specifies when the guardrail runs. | Set to `[pre_call, post_call]` to scan prompts before sending to the LLM and scan responses after receiving from the LLM. `during_call` and `logging_only` are also supported. |

### Optional Parameters

| Parameter | Description | Use |
|-----------|-------------|-----|
| `app_name` | Name of the application for policy assignment, tracking, and identification in the Vision One console. | Default: `litellm`. |
| `default_on` | Whether the guardrail runs on every request without requiring explicit user specification. | Default: `false`. |
| `fallback_on_error` | Policy for handling errors when making Guard API requests. | Specify `block` to reject requests on error, or `allow` to permit requests on error. Default: `block`. |
| `timeout` | Timeout for API requests, in seconds. | Default: `5.0`. |

### Example Configuration

Add the following to your `config.yaml` file:

```yaml
guardrails:
  - guardrail_name: "trendai-guard"
    litellm_params:
      guardrail: trendai_guard.TrendAIGuardrail                # Required
      mode: [pre_call, post_call]                              # Required
      api_base: https://api.xdr.trendmicro.com/v3.0/aiSecurity # Required; adjust URL for your deployment
      api_key: os.environ/TMV1_API_KEY                         # Required; set environment variable
      app_name: litellm                                        # Optional; customize for your application
      default_on: true                                         # Optional; enable by default
      fallback_on_error: block                                 # Optional; block requests on API errors
      timeout: 5.0                                             # Optional; request timeout in seconds
```

## Deploy the LiteLLM Guardrail

Ensure that the `trendai_guard.py` module is in the same directory as your `config.yaml` file, or provide the correct relative path in the `guardrail` parameter.

### Docker Deployment

Run the following command to deploy LiteLLM with the TrendAI guardrail:

```bash
docker run -d \
  -p 4000:4000 \
  -e TMV1_API_KEY=$TMV1_API_KEY \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  -v $(pwd)/trendai_guard.py:/app/trendai_guard.py \
  docker.litellm.ai/berriai/litellm:main-stable \
  --config /app/config.yaml --detailed_debug
```

**Environment variables:**
- `TMV1_API_KEY` — API key obtained in the prerequisites.

**Note:** LiteLLM supports multiple LLM providers and various methods for configuring model credentials, including the LiteLLM UI and API. The example above does not include LLM provider credentials. Add the environment variables required by your LLM provider (for example, `OPENAI_API_KEY` for OpenAI) to the `docker run` command. For more information, see the [LiteLLM documentation](https://docs.litellm.ai/docs/).

### Helm Deployment

**Supported versions:** LiteLLM Helm chart version `1.81.6` or later (`1.83.10` or later recommended).

#### Before You Start

Prepare a custom values file for your LiteLLM Helm release that includes the guardrail parameters in the `proxy_config` section. For configuration details, refer to the [LiteLLM Helm Chart README](https://github.com/BerriAI/litellm/blob/main/deploy/charts/litellm-helm/README.md).

For the `api_key` parameter, populate the `TMV1_API_KEY` environment variable from a Kubernetes Secret.

#### Create and Configure the Kubernetes Secret

1. Create a secret named `tmv1-api-key` with your API key:

```bash
kubectl create secret generic tmv1-api-key \
  --from-literal=TMV1_API_KEY=<YOUR_API_KEY> \
  --namespace <LITELLM_NAMESPACE> \
  --create-namespace
```

Replace `<YOUR_API_KEY>` with the API key obtained in the prerequisites, and `<LITELLM_NAMESPACE>` with the Kubernetes namespace where LiteLLM is deployed.

2. Add the secret to your custom values file:

```yaml
environmentSecrets:
  - tmv1-api-key
```

#### Install the Helm Chart

Run the following command to install the LiteLLM Helm chart with the guardrail configuration and latest released guardrail:

```bash
helm install <RELEASE_NAME> \
  oci://docker.litellm.ai/berriai/litellm-helm:<HELM_CHART_VERSION> \
  -f <YOUR_VALUES_FILE> \
  -f https://github.com/trendmicro/tm-v1-ai-guard-litellm-plugin/releases/latest/download/overrides.yaml \
  --namespace <LITELLM_NAMESPACE> \
  --create-namespace
```

You can also pin the guardrail to a specific version (e.g., `v1.0.0`):

```bash
helm install <RELEASE_NAME> \
  oci://docker.litellm.ai/berriai/litellm-helm:<HELM_CHART_VERSION> \
  -f <YOUR_VALUES_FILE> \
  -f https://github.com/trendmicro/tm-v1-ai-guard-litellm-plugin/releases/download/v1.0.0/overrides.yaml \
  --namespace <LITELLM_NAMESPACE> \
  --create-namespace
```

Replace:
- `<RELEASE_NAME>` with a descriptive name for your Helm release.
- `<HELM_CHART_VERSION>` with the LiteLLM Helm chart version (e.g., `1.83.10`).
- `<YOUR_VALUES_FILE>` with the path to your custom values file.
- `<LITELLM_NAMESPACE>` with your Kubernetes namespace.

**Custom volume configuration:** If your deployment already customizes `volumes`, `volumeMounts`, or `extraInitContainers`, merge the entries from `overrides.yaml` into your values file instead of using the `overrides.yaml` file directly.

## Verify the Deployment

### Confirm Guardrail Registration

1. Open the [LiteLLM Admin UI](https://docs.litellm.ai/docs/proxy/ui).
2. Navigate to **Guardrails** > **Guardrails** tab.
3. Confirm that `trendai-guard` is listed and shows the expected mode and default status.

### Test the Guardrail

1. In the LiteLLM Admin UI, navigate to **Guardrails** > **Test Playground**.
2. Enter a prompt designed to trigger the content scanners configured in your AI Guard policy in Vision One.
3. Verify that the guardrail blocks or allows the prompt according to your policy configuration.

**Expected behavior:**
- Pre-call mode blocks prompts that violate guardrail policies.
- Post-call mode blocks responses that violate guardrail policies.
- If an API error occurs and `fallback_on_error` is set to `block`, requests are rejected. If set to `allow`, requests proceed.
