# TrendAI LiteLLM Guardrail

LiteLLM Guardrail integration with the [AI Guard API](https://docs.trendmicro.com/en-us/documentation/article/trend-vision-one-ai-guard-api-reference)

## Prerequisites

### Obtaining an API Key

Navigate to the **Workflow and Automation** -> **Third-Party Integrations** -> **LiteLLM** page in the Vision One console and click the "Generate Token" button to obtain an API key.

Note: the account used to create the API key should be in the V1 region that corresponds to where your LiteLLM deployment is located.

### Obtaining the AI Guard Base URL

#### TrendAI Hosted
Navigate to the LiteLLM third party integrations page in the Vision One console and copy the Endpoint URL.

#### Self-hosted AWS
In the AWS management console open the AI Guard CloudFormation stack and copy the `GuardAPIEndpoint` in the **Outputs** tab.

#### Self-hosted Kubernetes
Use the internal Kubernetes Service URL exposed by the AI Guard Helm release.

If LiteLLM is deployed in the same namespace as AI Guard, set `api_base` to:

`http://<ai-guard-service-name>:8080`

If LiteLLM is deployed in a different namespace, use the fully qualified in-cluster DNS name:

`http://<ai-guard-service-name>.<namespace>.svc.cluster.local:8080`

For example, with the default Helm install of the AI Guard Helm chart, the service is typically reachable as:

`http://ai-guard:8080` (same namespace)

or

`http://ai-guard.trend-ai-security.svc.cluster.local:8080` (different namespace)

## Configuration

### Required Parameters
- `guardrail`: Set to `trendai_guard.TrendAIGuardrail`
- `api_base`: Base URL for the AI Guard API (e.g. `https://api.xdr.trendmicro.com/v3.0/aiSecurity`, `https://api.eu.xdr.trendmicro.com/v3.0/aiSecurity`).
- `api_key`: API key for authenticating with the AI Guard API. It is recommended to set this to an environment variable (e.g. `os.environ/TMV1_API_KEY`)
- `app_name`: Name of the application for tracking in the Vision One console (default: `litellm`)

### Optional Parameters
- `fallback_on_error`: Policy for handling errors when making Guard API requests (`block` or `allow`, default: `block`)
- `timeout`: Timeout for API requests (default: 5.0 seconds)

### Example `config.yaml`
```yaml
guardrails:
  - guardrail_name: "trendai-guard"
    litellm_params:
      guardrail: trendai_guard.TrendAIGuardrail
      mode: [pre_call, post_call]
      default_on: true # run on every request without needing the user to specify it
      api_base: http://ai-guard:8080
      api_key: os.environ/TMV1_API_KEY
      app_name: litellm
      fallback_on_error: block
      timeout: 5.0
```

## Deployment

Ensure that `trendai_guard.py` is located in the same directory as your `config.yaml` file or provide the appropriate relative path to the module in the `guardrail` parameter.

### Docker Run

```bash
docker run -d \
  -p 4000:4000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e TMV1_API_KEY=$TMV1_API_KEY \
  -v $(pwd)/my_config.yaml:/app/config.yaml \
  -v $(pwd)/trendai_guard.py:/app/trendai_guard.py \
  docker.litellm.ai/berriai/litellm:main-stable \
  --config /app/config.yaml --detailed_debug
```

### Helm
For convenience, we provide an overrides file for the LiteLLM Helm chart that downloads and mounts the `trendai_guard.py` module to the config directory via `extraInitContainers`.

LiteLLM Helm chart version `1.81.6+` required (`1.83.10+` recommended).

Before installing, ensure you have a custom values file for your LiteLLM release configured with the above guardrail parameters in `proxy_config`. For details on configuring the values file, refer to the [LiteLLM Helm Chart documentation](https://github.com/BerriAI/litellm/blob/main/deploy/charts/litellm-helm/README.md).

For the `api_key` parameter, populate the `TMV1_API_KEY` environment variable from a Kubernetes Secret:

Create a `tmv1-api-key` secret and populate it with a `TMV1_API_KEY` entry:
```bash
kubectl create secret generic tmv1-api-key --from-literal=TMV1_API_KEY=<YOUR_API_KEY> --namespace <LITELLM_NAMESPACE> --create-namespace
```

Add the secret to your LiteLLM values file with `environmentSecrets`:
```yaml
environmentSecrets:
  - tmv1-api-key
```


Install the LiteLLM Helm chart with your custom values file and the AI Guard overrides file.

To use the latest release:
```bash
helm install <RELEASE_NAME> \
  oci://docker.litellm.ai/berriai/litellm-helm:<HELM_CHART_VERSION> \
  -f <YOUR_VALUES_FILE> \
  -f https://github.com/trendmicro/tm-v1-ai-guard-litellm-plugin/releases/latest/download/overrides.yaml \
  --namespace <LITELLM_NAMESPACE> \
  --create-namespace
```

To pin to a specific version (e.g. `v1.0.0`):
```bash
helm install <RELEASE_NAME> \
  oci://docker.litellm.ai/berriai/litellm-helm:<HELM_CHART_VERSION> \
  -f <YOUR_VALUES_FILE> \
  -f https://github.com/trendmicro/tm-v1-ai-guard-litellm-plugin/releases/download/v1.0.0/overrides.yaml \
  --namespace <LITELLM_NAMESPACE> \
  --create-namespace
```

Note: if your deployment already customizes `volumes`, `volumeMounts`, or `extraInitContainers`, you will need to merge the AI Guard overrides entries into your own values file instead of using the `overrides.yaml` file directly.

## Verification

### Admin UI
1. From the [LiteLLM Admin UI](https://docs.litellm.ai/docs/proxy/ui), navigate to the `Guardrails` page and confirm that the `trendai-guard` guardrail is listed in the `Guardrails` tab and configured as expected.
2. Navigate to the `Test Playground` tab to test the guardrail against sample prompts.
