"""Azure OpenAI provider against a fake Azure endpoint (real OpenAI SDK, real HTTP layer, no network)."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from backend.core.config import Settings
from backend.core.exceptions import AgentOutputInvalid, ModelError
from backend.harness.model_gateway import AzureOpenAIProvider, ModelGateway, ModelTier

KEY = "azure-key-0123456789abcdef"


class Verdict(BaseModel):
    answer: str
    score: float


class Loose(BaseModel):  # dict[str, Any] cannot be expressed in strict mode -> non-strict + local validation
    answer: str
    extra: dict[str, Any] = {}


def settings(**overrides: Any) -> Settings:
    base = {"llm_provider": "azure_openai", "AZURE_OPENAI_ENDPOINT": "https://res.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": KEY, "AZURE_OPENAI_DEPLOYMENT": "gpt-4.1", "llm_max_retries": 0}
    base.update(overrides)
    return Settings(**base)


def responses_body(payload: dict[str, Any], model: str = "gpt-4.1") -> dict[str, Any]:
    return {"id": "resp_1", "object": "response", "created_at": 0, "model": model, "status": "completed",
            "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}]}],
            "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": []}


def chat_body(content: str, finish: str = "stop") -> dict[str, Any]:
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-4.1",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "refusal": None}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 7, "total_tokens": 27}}


class FakeAzure:
    def __init__(self, handler: Any) -> None:
        self.requests: list[httpx.Request] = []
        self.handler = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def provider(self, s: Settings) -> AzureOpenAIProvider:
        return AzureOpenAIProvider(s, http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)))


async def call(p: AzureOpenAIProvider, model: type[BaseModel] = Verdict, deployment: str = "gpt-4.1",
               images: list[Any] | None = None) -> Any:
    return await p.structured(model=deployment, system="You judge.", text="Is it ok?", images=images or [],
                              output_model=model, max_tokens=200)


async def test_responses_api_structured_output():
    fake = FakeAzure(lambda r: httpx.Response(200, json=responses_body({"answer": "ok", "score": 0.9})))
    out, usage = await call(fake.provider(settings()))
    assert out == Verdict(answer="ok", score=0.9)
    assert (usage.input_tokens, usage.output_tokens, usage.provider) == (12, 5, "azure_openai")
    req = fake.requests[0]
    assert req.url.path == "/openai/responses" and req.url.params["api-version"] == "2025-03-01-preview"
    assert req.headers["api-key"] == KEY
    body = json.loads(req.content)
    assert body["model"] == "gpt-4.1" and body["instructions"] == "You judge." and body["max_output_tokens"] == 200
    assert body["text"]["format"]["type"] == "json_schema" and body["text"]["format"]["strict"] is True


async def test_auto_mode_falls_back_to_chat_completions_and_remembers():
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path.endswith("/responses"):
            return httpx.Response(404, json={"error": {"code": "404", "message": "Resource not found"}})
        return httpx.Response(200, json=chat_body(json.dumps({"answer": "fallback", "score": 0.5})))

    fake = FakeAzure(handler)
    p = fake.provider(settings())
    out, usage = await call(p)
    assert out.answer == "fallback" and usage.input_tokens == 20
    assert fake.requests[1].url.path == "/openai/deployments/gpt-4.1/chat/completions"
    body = json.loads(fake.requests[1].content)
    assert body["response_format"]["type"] == "json_schema" and body["max_completion_tokens"] == 200
    await call(p)  # the decision is remembered: no second probe of /responses
    assert [r.url.path for r in fake.requests].count("/openai/responses") == 1


async def test_non_strict_schema_is_validated_locally():
    fake = FakeAzure(lambda r: httpx.Response(200, json=responses_body({"answer": "x", "extra": {"k": 1}})))
    out, _ = await call(fake.provider(settings()), model=Loose)
    assert out.extra == {"k": 1}
    assert json.loads(fake.requests[0].content)["text"]["format"]["strict"] is False


async def test_invalid_output_is_rejected_not_passed_on():
    fake = FakeAzure(lambda r: httpx.Response(200, json=responses_body({"answer": "x"})))  # score missing
    with pytest.raises((AgentOutputInvalid, Exception)):
        await call(fake.provider(settings(AZURE_OPENAI_API="chat")))


async def test_truncated_chat_output_is_reported():
    fake = FakeAzure(lambda r: httpx.Response(200, json=chat_body('{"answer": "x"', finish="length")))
    with pytest.raises(AgentOutputInvalid):
        await call(fake.provider(settings(AZURE_OPENAI_API="chat")))


@pytest.mark.parametrize("status,needle", [(401, "AZURE_OPENAI_API_KEY"), (404, "AZURE_OPENAI_DEPLOYMENT"),
                                           (429, "rate limit"), (403, "denied")])
async def test_errors_are_actionable_and_never_contain_the_key(status, needle):
    fake = FakeAzure(lambda r: httpx.Response(status, json={"error": {"code": str(status), "message": f"boom {KEY}"}}))
    with pytest.raises(ModelError) as info:
        await call(fake.provider(settings(AZURE_OPENAI_API="responses")))
    assert needle in str(info.value) and KEY not in str(info.value)


async def test_content_filter_is_named():
    fake = FakeAzure(lambda r: httpx.Response(400, json={"error": {"code": "content_filter",
                                                                   "message": "The response was filtered"}}))
    with pytest.raises(ModelError, match="content filter"):
        await call(fake.provider(settings(AZURE_OPENAI_API="responses")))


async def test_images_are_sent_as_data_urls():
    from backend.harness.model_gateway import ImageInput

    fake = FakeAzure(lambda r: httpx.Response(200, json=responses_body({"answer": "ok", "score": 1})))
    await call(fake.provider(settings()), images=[ImageInput(b"\x89PNG", "image/png")])
    content = json.loads(fake.requests[0].content)["input"][0]["content"]
    assert content[1]["type"] == "input_image" and content[1]["image_url"].startswith("data:image/png;base64,")


def test_each_tier_gets_its_own_deployment():
    s = settings(AZURE_OPENAI_CLASSIFICATION_DEPLOYMENT="gpt-4.1-mini", AZURE_OPENAI_VISION_DEPLOYMENT="gpt-4o")
    gw = ModelGateway(s)
    assert [m for _, m in gw.router.route(ModelTier.REASONING)] == ["gpt-4.1"]
    assert [m for _, m in gw.router.route(ModelTier.VISION)] == ["gpt-4o"]
    # a failed small-model classification escalates to the reasoning deployment
    assert [m for _, m in gw.router.route(ModelTier.CLASSIFICATION)] == ["gpt-4.1-mini", "gpt-4.1"]


def test_llm_model_settings_are_honoured_as_deployment_names():
    s = settings(llm_classification_model="gpt-4o-mini")  # AITEST_LLM_CLASSIFICATION_MODEL
    assert ModelGateway(s).router.route(ModelTier.CLASSIFICATION)[0][1] == "gpt-4o-mini"
    # the Claude defaults are not Azure deployments: those tiers use AZURE_OPENAI_DEPLOYMENT
    assert ModelGateway(s).router.route(ModelTier.REASONING)[0][1] == "gpt-4.1"


def test_unconfigured_azure_is_unavailable():
    s = settings(AZURE_OPENAI_ENDPOINT="", AZURE_OPENAI_DEPLOYMENT="")
    assert not ModelGateway(s).available


async def test_gateway_end_to_end_redacts_secrets_in_prompts():
    from backend.agents.governance.redaction import register_sensitive

    register_sensitive("hunter2-secret")
    fake = FakeAzure(lambda r: httpx.Response(200, json=responses_body({"answer": "ok", "score": 1})))
    s = settings()
    gw = ModelGateway(s, providers={"azure_openai": fake.provider(s)})
    result = await gw.structured(tier=ModelTier.REASONING, system="sys", prompt="password is hunter2-secret",
                                 output_model=Verdict, agent="test")
    assert result.output.answer == "ok" and result.attempts == ["azure_openai:gpt-4.1"]
    assert "hunter2-secret" not in fake.requests[0].content.decode()
