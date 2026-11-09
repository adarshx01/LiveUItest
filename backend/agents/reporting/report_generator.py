"""Evidence-backed execution reports (spec section 24).

Produces ``report.json`` (complete structured result) and a self-contained
``report.html``: every step with its status, the instruction, the normalized
action, the grounded target with confidence and locator strategy, memory use,
the Playwright code that ran, verification checks, assertions (expected vs
actual + evidence), recovery attempts, and before/after screenshots with the
acted-on element highlighted. Screenshots are embedded so the file can be
shared as-is; the execution video is linked alongside.
"""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import time
from typing import Any

from jinja2 import Environment, select_autoescape
from PIL import Image

from backend.agents.governance.redaction import redact_html, redact_value
from backend.core.logging import get_logger
from backend.dsl.models import TestSpec
from backend.dsl.results import ExecutionMetrics, StepResult, StepStatus
from backend.services.artifact_service import ArtifactService, get_artifact_service

log = get_logger(__name__)

STATUS_COLORS = {"PASSED": "#138a4a", "FAILED": "#c62828", "BLOCKED": "#b26a00", "INCONCLUSIVE": "#5f6b7a",
                 "SKIPPED": "#8a94a6", "RUNNING": "#3355ff", "PENDING": "#8a94a6"}

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ r.test_name }} — {{ r.outcome }}</title>
<style>
:root{--bg:#f5f6fa;--card:#fff;--ink:#1b2130;--muted:#5d6779;--line:#e2e5ee;--pass:#138a4a;--fail:#c62828;--block:#b26a00;--inc:#5f6b7a}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--ink)}
header{background:#141a2b;color:#fff;padding:26px 32px}header h1{margin:0 0 6px;font-size:22px}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:600;font-size:12px;color:#fff;letter-spacing:.3px}
.big{font-size:15px;padding:6px 16px}main{max-width:1180px;margin:0 auto;padding:22px 20px 60px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.metric .v{font-size:22px;font-weight:700}.metric .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:4px 24px;font-size:13px}
.meta div span{color:var(--muted)}h2{font-size:17px;margin:28px 0 10px}
.step{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--line);border-radius:10px;margin:12px 0;overflow:hidden}
.step summary{cursor:pointer;padding:12px 16px;display:flex;gap:12px;align-items:center;list-style:none}
.step summary::-webkit-details-marker{display:none}.idx{font-weight:700;color:var(--muted);min-width:28px}
.instr{flex:1}.instr .norm{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.body{padding:4px 18px 18px;border-top:1px solid var(--line)}
.kv{display:grid;grid-template-columns:170px 1fr;gap:4px 14px;font-size:13px;margin:10px 0}.kv .k{color:var(--muted)}
.shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px;margin-top:10px}
.shot figure{margin:0}.shot img{width:100%;border:1px solid var(--line);border-radius:8px;cursor:zoom-in}
.shot figcaption{font-size:12px;color:var(--muted);margin-top:4px}
code,pre{font-family:ui-monospace,Menlo,monospace;font-size:12px}pre{background:#0f1422;color:#d7def0;padding:10px 12px;border-radius:8px;overflow:auto;max-height:180px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
.ok{color:var(--pass);font-weight:600}.bad{color:var(--fail);font-weight:600}.warn{color:var(--block);font-weight:600}
.pill{display:inline-block;background:#eef1f8;border-radius:6px;padding:1px 7px;margin:1px 2px;font-size:12px}
.lightbox{position:fixed;inset:0;background:rgba(10,14,24,.88);display:none;align-items:center;justify-content:center;z-index:9}
.lightbox img{max-width:94vw;max-height:92vh;border-radius:6px}.lightbox.open{display:flex}
.note{color:var(--muted);font-size:12px}video{width:100%;max-width:900px;border-radius:8px;border:1px solid var(--line)}
</style></head><body>
<header>
  <div style="display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center">
    <div><h1>{{ r.test_name }}</h1><div style="opacity:.8">{{ r.reason }}</div></div>
    <span class="badge big" style="background:{{ colors[r.outcome] }}">{{ r.outcome }}</span>
  </div>
</header>
<main>
<div class="card meta">
  <div><span>Execution ID:</span> <code>{{ r.execution_id }}</code></div>
  <div><span>Test case:</span> <code>{{ r.test_case_id }}</code> v{{ r.test_case_version }}</div>
  <div><span>Application:</span> {{ r.application }}</div>
  <div><span>Environment:</span> {{ r.environment }}</div>
  <div><span>Browser:</span> {{ r.browser }}</div>
  <div><span>Runner:</span> {{ r.runner }}</div>
  {% if r.scope %}<div><span>Scope:</span> <span class="warn">partial re-run</span> of steps {{ r.scope.selected|join(", ") }}{% if r.scope.rerun_of %} (from {{ r.scope.rerun_of }}){% endif %}{% if r.scope.prerequisites %}; {{ r.scope.prerequisites|length }} prerequisite step(s) re-run{% endif %}{% if r.scope.not_run %}; {{ r.scope.not_run|length }} step(s) not run{% endif %}</div>{% endif %}
  <div><span>Started:</span> {{ r.started }}</div>
  <div><span>Duration:</span> {{ "%.1f"|format(r.metrics.duration_ms / 1000) }} s</div>
  <div><span>Original strategy:</span> {% if r.original_strategy_succeeded %}<span class="ok">succeeded without recovery</span>{% else %}<span class="warn">{{ r.metrics.recovered_steps }} step(s) required recovery</span>{% endif %}</div>
</div>
<div class="grid">
  {% for k, v, c in cards %}<div class="card metric"><div class="v" style="color:{{ c }}">{{ v }}</div><div class="k">{{ k }}</div></div>{% endfor %}
</div>
{% if r.video %}<h2>Execution video</h2><video controls src="{{ r.video }}"></video><div class="note">Recorded by Playwright MCP for the whole isolated browser session.</div>{% endif %}
{% if r.replay %}<h2>Execution replay</h2><img src="{{ r.replay }}" alt="execution replay" style="max-width:100%;border-radius:8px;border:1px solid var(--line)"><div class="note">Animated replay of every action (before / after screenshots).</div>{% endif %}
<h2>Step-by-step execution</h2>
{% for s in steps %}
<details class="step" style="border-left-color:{{ colors[s.status] }}" {% if s.status in ('FAILED','BLOCKED','INCONCLUSIVE') %}open{% endif %}>
  <summary>
    <span class="idx">{{ s.index }}.</span>
    <span class="instr">{{ s.original_text }}<div class="norm">{{ s.summary }}</div></span>
    {% if s.recovered %}<span class="pill">recovered</span>{% endif %}{% if s.memory_used %}<span class="pill">memory</span>{% endif %}
    <span class="note">{{ "%.0f"|format(s.duration_ms) }} ms</span>
    <span class="badge" style="background:{{ colors[s.status] }}">{{ s.status }}</span>
  </summary>
  <div class="body">
    {% if s.message %}<p><strong>Result:</strong> {{ s.message }}</p>{% endif %}
    {% for a in s.actions %}
      <div class="kv">
        <div class="k">Attempt</div><div>#{{ a.attempt }} — {{ a.action }} {% if a.ok %}<span class="ok">tool call ok</span>{% else %}<span class="bad">tool call failed</span>{% endif %}{% if a.failure_category %} <span class="pill">{{ a.failure_category }}</span>{% endif %}</div>
        {% if a.grounding %}
        <div class="k">Target</div><div>{{ a.grounding.display }} <span class="pill">confidence {{ "%.2f"|format(a.grounding.confidence) }}</span><span class="pill">{{ a.grounding.decision }}</span><span class="pill">{{ a.grounding.method }}</span></div>
        {% if a.strategy %}<div class="k">Locator strategy</div><div>{% for k, v in a.strategy.items() %}<span class="pill">{{ k }}: {{ v }}</span>{% endfor %}</div>{% endif %}
        {% if a.grounding.evidence %}<div class="k">Evidence</div><div>{% for e in a.grounding.evidence[:5] %}<span class="pill">{{ e }}</span>{% endfor %}</div>{% endif %}
        {% if a.candidates %}<div class="k">Candidates</div><div>{% for c in a.candidates %}<span class="pill">{{ c.display }} · {{ "%.2f"|format(c.score) }}</span>{% endfor %}</div>{% endif %}
        {% endif %}
        {% if a.tool %}<div class="k">Tool</div><div><code>{{ a.tool }}</code> {% if a.args %}<code>{{ a.args }}</code>{% endif %}</div>{% endif %}
        {% if a.error %}<div class="k">Error</div><div class="bad">{{ a.error }}</div>{% endif %}
        {% if a.verification %}<div class="k">Verification</div><div>{{ a.verification.status }} — {{ a.verification.message }}{% for c in a.verification.checks %} <span class="pill">{{ c.check }}: {{ 'ok' if c.ok else 'no' }}</span>{% endfor %}</div>{% endif %}
      </div>
      {% if a.playwright_code %}<pre>{{ a.playwright_code }}</pre>{% endif %}
      {% if a.before or a.after %}
      <div class="shots">
        {% if a.before %}<div class="shot"><figure><img src="{{ a.before }}" alt="before"><figcaption>Before action (target highlighted)</figcaption></figure></div>{% endif %}
        {% if a.after %}<div class="shot"><figure><img src="{{ a.after }}" alt="after"><figcaption>After action</figcaption></figure></div>{% endif %}
      </div>
      {% endif %}
    {% endfor %}
    {% if s.assertions %}
      <h4>Assertions</h4>
      <table><tr><th>Assertion</th><th>Expected</th><th>Actual</th><th>Status</th><th>Evidence</th></tr>
      {% for x in s.assertions %}<tr><td>{{ x.type }}<div class="note">{{ x.description }}</div></td><td><code>{{ x.expected }}</code>{% if x.derived_expectation %}<div class="note">derived from test data</div>{% endif %}</td><td><code>{{ x.actual }}</code></td>
      <td class="{{ 'ok' if x.status == 'PASSED' else 'bad' }}">{{ x.status }}<div class="note">{{ x.message }}</div></td><td>{% for e in x.evidence %}<span class="pill">{{ e.source }}</span>{% endfor %}</td></tr>{% endfor %}
      </table>
    {% endif %}
    {% if s.recovery %}
      <h4>Recovery</h4>
      <table><tr><th>#</th><th>Failure</th><th>Strategy</th><th>Rationale</th><th>Outcome</th></tr>
      {% for x in s.recovery %}<tr><td>{{ x.attempt }}</td><td>{{ x.failure_category }}</td><td><code>{{ x.strategy }}</code></td><td>{{ x.rationale }}<div class="note">{{ x.details.detail }}</div></td><td class="{{ 'ok' if x.outcome == 'recovered' else 'bad' }}">{{ x.outcome }}</td></tr>{% endfor %}
      </table>
    {% endif %}
    {% if s.memory_details %}<div class="note">Memory: {{ s.memory_details }}</div>{% endif %}
  </div>
</details>
{% endfor %}
{% if r.security_warnings %}<h2>Security warnings</h2><div class="card">{% for w in r.security_warnings %}<div>⚠ {{ w }}</div>{% endfor %}<div class="note">Page content is untrusted: instruction-like text was recorded and ignored.</div></div>{% endif %}
{% if r.plan_issues %}<h2>Plan analysis</h2><div class="card"><table><tr><th>Severity</th><th>Code</th><th>Message</th></tr>{% for i in r.plan_issues %}<tr><td>{{ i.severity }}</td><td>{{ i.code }}</td><td>{{ i.message }}</td></tr>{% endfor %}</table></div>{% endif %}
{% if r.metrics.llm_calls %}<h2>Model usage</h2><div class="card"><table><tr><th>Agent</th><th>Calls</th><th>Failed</th><th>Input tokens</th><th>Cached</th><th>Output tokens</th><th>Reasoning</th><th>Cost</th><th>Latency</th></tr>{% for name, u in r.metrics.llm_usage_by_agent.items() %}<tr><td>{{ name }}</td><td>{{ u.calls|int }}</td><td>{{ u.failed|int }}</td><td>{{ "{:,}".format(u.input_tokens|int) }}</td><td>{{ "{:,}".format(u.cached_tokens|int) }}</td><td>{{ "{:,}".format(u.output_tokens|int) }}</td><td>{{ "{:,}".format(u.reasoning_tokens|int) }}</td><td>${{ "%.4f"|format(u.cost_usd) }}</td><td>{{ "%.1f"|format(u.latency_ms / 1000) }} s</td></tr>{% endfor %}<tr><th>Total</th><th>{{ r.metrics.llm_calls }}</th><th>{{ r.metrics.llm_failed_calls }}</th><th>{{ "{:,}".format(r.metrics.llm_input_tokens) }}</th><th>{{ "{:,}".format(r.metrics.llm_cached_tokens) }}</th><th>{{ "{:,}".format(r.metrics.llm_output_tokens) }}</th><th>{{ "{:,}".format(r.metrics.llm_reasoning_tokens) }}</th><th>${{ "%.4f"|format(r.metrics.llm_cost_usd) }}</th><th>{{ "%.1f"|format(r.metrics.model_latency_ms / 1000) }} s</th></tr></table><div class="note">{% for name, u in r.metrics.llm_usage_by_model.items() %}{{ name }}: {{ u.calls|int }} calls, {{ "{:,}".format((u.input_tokens + u.output_tokens)|int) }} tokens{% if not loop.last %} · {% endif %}{% endfor %}. Cost uses AITEST_LLM_PRICE_* (per million tokens).</div></div>{% endif %}
<h2>Artifacts</h2><div class="card">{% for a in r.artifacts %}<div><span class="pill">{{ a.kind }}</span> <a href="{{ a.href }}">{{ a.name }}</a></div>{% endfor %}</div>
<h2>Execution timeline</h2><div class="card"><table><tr><th>t (s)</th><th>Event</th><th>Step</th><th>Message</th></tr>{% for e in r.timeline %}<tr><td>{{ e.t }}</td><td><code>{{ e.event }}</code></td><td>{{ e.step_id or '' }}</td><td>{{ e.message }}</td></tr>{% endfor %}</table></div>
<p class="note">Generated {{ r.generated }} by the AI testing platform. Secrets and sensitive values are redacted.</p>
</main>
<div class="lightbox" id="lb" onclick="this.classList.remove('open')"><img id="lbimg" alt=""></div>
<script>document.querySelectorAll('.shot img').forEach(function(i){i.addEventListener('click',function(){document.getElementById('lbimg').src=i.src;document.getElementById('lb').classList.add('open');});});</script>
</body></html>"""


def _jpeg_data_uri(png: bytes, max_width: int = 1280) -> str:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    if image.width > max_width:
        image = image.resize((max_width, int(image.height * max_width / image.width)))
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


class ReportGenerator:
    name = "report_generator_agent"

    def __init__(self, artifacts: ArtifactService | None = None) -> None:
        self.artifacts = artifacts or get_artifact_service()
        self.env = Environment(autoescape=select_autoescape(["html"]))

    async def _image(self, uri: str | None) -> str | None:
        if not uri:
            return None
        try:
            return _jpeg_data_uri(await self.artifacts.get_bytes(uri))
        except Exception:  # pragma: no cover - missing artifact
            return None

    async def generate(self, state: dict[str, Any], runtime: Any) -> dict[str, Any]:
        execution_id = state["execution_id"]
        spec = TestSpec.model_validate(state["test_spec"]) if state.get("test_spec") else None
        final = state.get("final_result") or {"outcome": "BLOCKED", "reason": state.get("blocked_reason") or ""}
        steps: list[StepResult] = []
        if runtime is not None and spec is not None:
            for s in spec.flat_steps():
                if s.id in runtime.results:
                    steps.append(runtime.results[s.id])
        metrics = self._metrics(steps, runtime)
        events = await self._events(execution_id)
        report: dict[str, Any] = {
            "execution_id": execution_id,
            "test_case_id": state.get("test_case_id"),
            "test_case_version": state.get("test_case_version", 1),
            "test_name": spec.test_name if spec else (state.get("raw_input") or {}).get("name", "Test"),
            "outcome": final.get("outcome"),
            "reason": final.get("reason", ""),
            "original_strategy_succeeded": final.get("original_strategy_succeeded", True),
            "application": f"{state.get('application_id')} ({spec.base_url if spec else ''})",
            "environment": state.get("environment_id") or "default",
            "browser": self._browser(runtime),
            "runner": state.get("runner") or (runtime.runner if runtime else "n/a"),
            "started": dt.datetime.fromtimestamp(runtime.started, dt.UTC).isoformat() if runtime else "",
            "generated": dt.datetime.now(dt.UTC).isoformat(),
            "metrics": metrics.model_dump(),
            "plan_issues": state.get("plan_issues") or [],
            "security_warnings": runtime.security_warnings if runtime else [],
            "learning": state.get("learning_stats") or {},
            "steps": [s.model_dump(mode="json") for s in steps],
            "scope": state.get("step_scope"),
            "video": None,
        }
        report = redact_value(report)
        json_ref = await self.artifacts.put_bytes(execution_id, "report", "report.json",
                                                  json.dumps(report, indent=2, default=str).encode(), "application/json")
        replay_ref = await self._replay(execution_id, steps)
        if replay_ref is not None:
            report["replay"] = replay_ref.uri
            if runtime is not None and runtime.recorder is not None:
                await runtime.recorder.artifact(replay_ref, None)
        html = redact_html(await self._html(report, steps, runtime, events, metrics))
        html_ref = await self.artifacts.put_bytes(execution_id, "report", "report.html", html.encode("utf-8"),
                                                  "text/html")
        if runtime is not None and runtime.recorder is not None:
            await runtime.recorder.artifact(json_ref, None)
            await runtime.recorder.artifact(html_ref, None)
        return {"html_uri": html_ref.uri, "json_uri": json_ref.uri, "html_api_path": html_ref.api_path,
                "json_api_path": json_ref.api_path, "outcome": report["outcome"],
                "video_uri": runtime.video_artifact if runtime else None,
                "local_path": str(self.artifacts.local_path(html_ref.uri) or "")}

    async def _replay(self, execution_id: str, steps: list[StepResult]) -> Any:
        """Animated execution replay from the per-action screenshots (always available)."""

        frames: list[Image.Image] = []
        durations: list[int] = []
        for step in steps:
            for action in step.actions:
                for uri, hold in ((action.screenshot_before, 1100), (action.screenshot_after, 1300)):
                    if not uri:
                        continue
                    try:
                        image = Image.open(io.BytesIO(await self.artifacts.get_bytes(uri))).convert("RGB")
                    except Exception:  # pragma: no cover
                        continue
                    if image.width > 960:
                        image = image.resize((960, int(image.height * 960 / image.width)))
                    frames.append(_caption(image, f"{step.index}. {step.original_text[:90]} — {step.status.value}"))
                    durations.append(hold)
        if len(frames) < 2:
            return None
        size = frames[0].size
        frames = [f if f.size == size else f.resize(size) for f in frames][:240]
        out = io.BytesIO()
        frames[0].save(out, format="WEBP", save_all=True, append_images=frames[1:], duration=durations[:len(frames)],
                       loop=0, quality=70, method=4)
        return await self.artifacts.put_bytes(execution_id, "replay", "execution-replay.webp", out.getvalue(),
                                              "image/webp")

    @staticmethod
    def _browser(runtime: Any) -> str:
        if runtime is None or runtime.session is None and not getattr(runtime, "closed", False):
            return "not started"
        settings = runtime.settings
        viewport = runtime.request.options.viewport or settings.viewport_size()
        return f"{runtime.request.options.browser or settings.browser_name} " \
               f"(Playwright MCP, {'headless' if settings.browser_headless else 'headed'}, {viewport[0]}x{viewport[1]})"

    @staticmethod
    def _metrics(steps: list[StepResult], runtime: Any) -> ExecutionMetrics:
        m = ExecutionMetrics(total_steps=len(steps))
        for s in steps:
            if s.status == StepStatus.PASSED:
                m.passed_steps += 1
            elif s.status == StepStatus.FAILED:
                m.failed_steps += 1
            elif s.status == StepStatus.BLOCKED:
                m.blocked_steps += 1
            elif s.status == StepStatus.SKIPPED:
                m.skipped_steps += 1
            elif s.status == StepStatus.INCONCLUSIVE:
                m.inconclusive_steps += 1
            if s.recovered:
                m.recovered_steps += 1
            for a in s.assertions:
                m.assertions_total += 1
                m.assertions_passed += a.status == "PASSED"
                m.assertions_failed += a.status == "FAILED"
        if runtime is not None:
            b = runtime.budget
            m.retries, m.recoveries = runtime.retries, runtime.recoveries
            m.memory_hits, m.memory_lookups = runtime.memory_hits, runtime.memory_lookups
            m.memory_fastpath = getattr(runtime, "memory_fastpath", 0)
            m.llm_calls, m.llm_input_tokens, m.llm_output_tokens = b.llm_calls, b.llm_input_tokens, b.llm_output_tokens
            m.llm_cost_usd = round(b.llm_cost_usd, 6)
            m.llm_cached_tokens, m.llm_reasoning_tokens = b.llm_cached_tokens, b.llm_reasoning_tokens
            m.llm_failed_calls = b.llm_failed_calls
            m.llm_usage_by_agent = {k: dict(v) for k, v in sorted(b.llm_by_agent.items())}
            m.llm_usage_by_model = {k: dict(v) for k, v in sorted(b.llm_by_model.items())}
            m.tool_calls = b.tool_calls
            m.model_latency_ms, m.tool_latency_ms = round(b.model_latency_ms, 1), round(b.tool_latency_ms, 1)
            m.duration_ms = round((time.time() - runtime.started) * 1000, 1)
        return m

    async def _events(self, execution_id: str) -> list[dict[str, Any]]:
        from backend.graph.services import get_graph_services

        services = get_graph_services()
        if services.sessionmaker is None:
            return []
        from sqlalchemy import select

        from backend.models.entities import ExecutionEvent

        try:
            async with services.sessionmaker() as session:
                rows = (await session.execute(select(ExecutionEvent).where(ExecutionEvent.execution_id == execution_id)
                                              .order_by(ExecutionEvent.seq))).scalars().all()
            return [{"ts": r.ts, "event": r.event, "step_id": r.step_id, "message": r.message} for r in rows]
        except Exception:  # pragma: no cover
            return []

    async def _html(self, report: dict[str, Any], steps: list[StepResult], runtime: Any,
                    events: list[dict[str, Any]], metrics: ExecutionMetrics) -> str:
        view_steps = []
        for s in steps:
            actions = []
            for a in s.actions:
                grounding = a.grounding
                strategy = {}
                if grounding and grounding.locator_strategy:
                    strategy = {k: v for k, v in grounding.locator_strategy.items()
                                if k in ("role", "name", "label", "placeholder", "test_id", "form_context",
                                         "dialog_context", "section_heading") and v}
                actions.append({
                    "attempt": a.attempt, "action": a.action, "ok": a.ok, "failure_category": a.failure_category,
                    "grounding": grounding, "strategy": redact_value(strategy),
                    "candidates": grounding.candidates[:4] if grounding else [],
                    "tool": a.tool, "args": json.dumps(redact_value(a.tool_args))[:300] if a.tool_args else "",
                    "error": a.error, "verification": a.verification or None,
                    "playwright_code": a.playwright_code,
                    "before": await self._image(a.screenshot_before), "after": await self._image(a.screenshot_after),
                })
            view_steps.append({"index": s.index, "original_text": s.original_text, "summary": s.summary,
                               "status": s.status.value, "message": s.message, "duration_ms": s.duration_ms,
                               "recovered": s.recovered, "memory_used": s.memory_used, "actions": actions,
                               "assertions": [a.model_dump() for a in s.assertions],
                               "recovery": [r.model_dump() for r in s.recovery],
                               "memory_details": s.memory_details.get("used") if s.memory_details else None})
        colors = STATUS_COLORS
        cards = [
            ("Total steps", metrics.total_steps, "#1b2130"), ("Passed", metrics.passed_steps, colors["PASSED"]),
            ("Failed", metrics.failed_steps, colors["FAILED"]), ("Blocked", metrics.blocked_steps, colors["BLOCKED"]),
            ("Skipped", metrics.skipped_steps, colors["SKIPPED"]), ("Recovered", metrics.recovered_steps, "#3355ff"),
            ("Assertions", f"{metrics.assertions_passed}/{metrics.assertions_total}", "#1b2130"),
            ("Retries", metrics.retries, "#1b2130"), ("Recoveries", metrics.recoveries, "#1b2130"),
            ("Memory reuse", f"{metrics.memory_hits}/{metrics.memory_lookups}", "#1b2130"),
            ("Direct replays", metrics.memory_fastpath, "#1b2130"),
            ("LLM calls", metrics.llm_calls, "#1b2130"),
            ("LLM tokens", f"{metrics.llm_input_tokens + metrics.llm_output_tokens:,}", "#1b2130"),
            ("Agent cost", f"${metrics.llm_cost_usd:.4f}", "#1b2130"),
            ("Tool calls", metrics.tool_calls, "#1b2130"),
        ]
        artifacts = [{"kind": "report", "name": "report.json", "href": "report.json"}]
        if report.get("replay"):
            artifacts.append({"kind": "replay", "name": "execution-replay.webp", "href": "../replay/execution-replay.webp"})
        if runtime is not None and runtime.video_artifact:
            artifacts.append({"kind": "video", "name": "execution-video.webm", "href": "../video/execution-video.webm"})
        for dl in (runtime.downloads if runtime else []):
            artifacts.append({"kind": "download", "name": dl.get("name"), "href": f"../downloads/{dl.get('name')}"})
        start = events[0]["ts"] if events else time.time()
        timeline = [{"t": f"{e['ts'] - start:.1f}", "event": e["event"], "step_id": e["step_id"],
                     "message": (e["message"] or "")[:160]} for e in events
                    if e["event"] not in ("PAGE_OBSERVED", "SCREENSHOT_CAPTURED")][:400]
        r = dict(report)
        r["metrics"] = metrics
        r["video"] = "../video/execution-video.webm" if runtime is not None and runtime.video_artifact else None
        r["replay"] = "../replay/execution-replay.webp" if report.get("replay") else None
        r["artifacts"] = artifacts
        r["timeline"] = timeline
        r["metrics"].recovered_steps = metrics.recovered_steps  # type: ignore[attr-defined]
        template = self.env.from_string(TEMPLATE)
        return template.render(r=_Obj(r), steps=view_steps, colors=colors, cards=cards)


def _caption(image: Image.Image, text: str) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    canvas = Image.new("RGB", (image.width, image.height + 28), (20, 26, 43))
    canvas.paste(image, (0, 28))
    ImageDraw.Draw(canvas).text((10, 8), text, fill=(255, 255, 255), font=ImageFont.load_default())
    return canvas


class _Obj(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
