"""Redaction rules (spec 16.4 / 28): secrets never leave the process unmasked, counters are never masked."""

from __future__ import annotations

import base64

import pytest

from backend.agents.governance import redaction
from backend.agents.governance.redaction import (
    MASK,
    is_sensitive_name,
    mask_values,
    redact_html,
    redact_value,
    register_sensitive,
)
from backend.browser.response_parser import ToolResponse
from backend.browser.tool_adapter import (
    _scrub_echoed_values,
    _sensitive_argument_values,
)


@pytest.fixture(autouse=True)
def isolated_registry():
    saved = set(redaction._sensitive_values)
    redaction._sensitive_values.clear()
    yield
    redaction._sensitive_values.clear()
    redaction._sensitive_values.update(saved)


@pytest.mark.parametrize("name", ["password", "test_password", "userPassword", "userpassword", "apiKey", "x-api-key",
                                  "X-Auth-Token", "access_token", "accesstoken", "session_id", "card_number", "cvv",
                                  "pin", "otp_code", "one_time_code", "Authorization", "set-cookie", "client_secret"])
def test_credential_names_are_sensitive(name):
    assert is_sensitive_name(name)


@pytest.mark.parametrize("name", ["passed_steps", "assertions_passed", "llm_input_tokens", "llm_output_tokens",
                                  "bypass", "spinner", "pinned", "tokenizer", "username", "display_name", "compass"])
def test_counters_and_ordinary_names_are_not(name):
    assert not is_sensitive_name(name)


def test_report_metrics_survive_redaction():
    metrics = {"passed_steps": 5, "assertions_passed": 2, "llm_input_tokens": 120, "llm_output_tokens": 30,
               "password": "hunter22", "pin": 1234, "cookie_banner_visible": True}
    out = redact_value(metrics)
    assert out["passed_steps"] == 5 and out["assertions_passed"] == 2
    assert out["llm_input_tokens"] == 120 and out["llm_output_tokens"] == 30
    assert out["password"] == MASK and out["pin"] == MASK
    assert out["cookie_banner_visible"] is True  # booleans carry no secret


def test_playwright_code_echo_is_scrubbed_at_the_adapter():
    """Playwright MCP reports the code it ran, including the typed value."""

    values = _sensitive_argument_values({"ref": "e7", "text": "bobpass", "submit": False}, ("text",))
    assert values == ["bobpass"]
    response = ToolResponse(tool="browser_type", ok=False,
                            code="await page.getByTestId('auth-secret').fill('bobpass');",
                            error="Timeout typing 'bobpass' into textbox", text="### Ran Playwright code\nfill('bobpass')",
                            sections={"Ran Playwright code": "fill('bobpass')"})
    _scrub_echoed_values(response, values)
    for text in (response.code, response.error, response.text, *response.sections.values()):
        assert "bobpass" not in text and MASK in text


def test_fill_form_values_are_extracted():
    fields = [{"name": "PIN", "type": "textbox", "ref": "e3", "value": "4321"},
              {"name": "Remember", "type": "checkbox", "ref": "e4", "value": "true"}]
    assert _sensitive_argument_values({"fields": fields}, ("fields",)) == ["4321", "true"]


def test_short_secrets_are_masked_only_when_quoted():
    assert mask_values("fill('ab') then cab", ["ab"]) == f"fill('{MASK}') then cab"
    assert mask_values("fill(&#39;ab&#39;)", ["ab"]) == f"fill(&#39;{MASK}&#39;)"


def test_html_pass_masks_escaped_values_and_keeps_images_intact():
    secret = "a&b<Secret>9"
    register_sensitive(secret)
    # an image whose base64 happens to contain registered text must not be altered
    register_sensitive("QUJD")
    image = "data:image/jpeg;base64," + base64.b64encode(b"ABC" * 40).decode()
    assert "QUJD" in image
    document = (f"<pre>fill(&#39;a&amp;b&lt;Secret&gt;9&#39;)</pre><img src='{image}'>"
                f"<p>raw a&b<Secret>9 and QUJD</p>")
    out = redact_html(document)
    assert "a&amp;b&lt;Secret&gt;9" not in out and "a&b<Secret>9" not in out
    assert image in out, "embedded screenshot was modified"
    assert "and QUJD" not in out
