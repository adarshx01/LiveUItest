"use client";
import { useState } from "react";
import { artifactUrl } from "@/services/api";
import type { StepResult } from "@/types/api";
import { StatusBadge, statusColor } from "./StatusBadge";

/** Intelligence view of one step: grounding evidence, candidates, strategy, memory, recovery, assertions. */
export function StepPanel({ step }: { step: StepResult }) {
  const [open, setOpen] = useState(step.status === "FAILED" || step.status === "BLOCKED");
  const last = step.actions[step.actions.length - 1];
  return (
    <div className="step" style={{ borderLeftColor: statusColor(step.status) }} onClick={() => setOpen(!open)}>
      <div className="row">
        <strong>{step.index}.</strong>
        <span style={{ flex: 1 }}>{step.original_text}<div className="muted">{step.summary}</div></span>
        {step.recovered && <span className="pill">recovered</span>}
        {step.memory_used && <span className="pill">memory</span>}
        <span className="muted">{Math.round(step.duration_ms)} ms</span>
        <StatusBadge status={step.status} />
      </div>
      {open && (
        <div onClick={(e) => e.stopPropagation()} style={{ marginTop: 8 }}>
          {step.message && <div><strong>Result:</strong> {step.message}</div>}
          {last?.grounding && (
            <table><tbody>
              <tr><th>Selected target</th><td>{last.grounding.display} <span className="pill">confidence {last.grounding.confidence.toFixed(2)}</span>
                <span className="pill">{last.grounding.decision}</span><span className="pill">{last.grounding.method}</span></td></tr>
              <tr><th>Locator strategy</th><td>{Object.entries(last.grounding.locator_strategy ?? {})
                .filter(([k]) => ["role", "name", "label", "test_id", "form_context", "placeholder"].includes(k))
                .map(([k, v]) => <span key={k} className="pill">{k}: {String(v)}</span>)}</td></tr>
              <tr><th>Why</th><td>{last.grounding.evidence.map((e) => <span key={e} className="pill">{e}</span>)}</td></tr>
              <tr><th>Candidates</th><td>{last.grounding.candidates.map((c) => <span key={c.display + c.score} className="pill">{c.display} · {c.score.toFixed(2)}</span>)}</td></tr>
            </tbody></table>
          )}
          {last?.playwright_code && <pre>{last.playwright_code}</pre>}
          {step.assertions.length > 0 && (
            <table><thead><tr><th>Assertion</th><th>Expected</th><th>Actual</th><th>Status</th></tr></thead><tbody>
              {step.assertions.map((a) => <tr key={a.assertion_id}><td>{a.type}</td><td><code>{JSON.stringify(a.expected)}</code></td>
                <td><code>{JSON.stringify(a.actual)?.slice(0, 120)}</code></td><td><StatusBadge status={a.status} /> <span className="muted">{a.message}</span></td></tr>)}
            </tbody></table>
          )}
          {step.recovery.length > 0 && (
            <table><thead><tr><th>Recovery</th><th>Failure</th><th>Rationale</th><th>Outcome</th></tr></thead><tbody>
              {step.recovery.map((r) => <tr key={r.attempt + r.strategy}><td><code>{r.strategy}</code></td><td>{r.failure_category}</td>
                <td>{r.rationale}</td><td>{r.outcome}</td></tr>)}
            </tbody></table>
          )}
          <div className="row" style={{ marginTop: 8 }}>
            {[last?.screenshot_before, last?.screenshot_after].map((uri, i) => {
              const src = artifactUrl(uri);
              return src ? <a key={i} href={src} target="_blank" rel="noreferrer"><img src={src} alt={i ? "after" : "before"} style={{ width: 300, borderRadius: 6, border: "1px solid #e1e5ee" }} /></a> : null;
            })}
          </div>
        </div>
      )}
    </div>
  );
}
