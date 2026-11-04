"use client";
import { use, useEffect, useState } from "react";
import { api } from "@/services/api";
import type { Comparison, ComparedStep } from "@/types/api";
import { StatusBadge } from "@/components/StatusBadge";

function Cell({ step }: { step: ComparedStep | null }) {
  if (!step) return <td className="muted">not run</td>;
  return (
    <td>
      <StatusBadge status={step.status} /> <span className="muted">{(step.duration_ms / 1000).toFixed(1)}s</span>
      {step.recovered && <span className="pill">recovered</span>}{step.memory_used && <span className="pill">memory</span>}
      {step.target?.display && <div className="muted"><code>{step.target.display}</code>{step.target.method && <> · {step.target.method}</>}</div>}
    </td>
  );
}

/** Execution comparison: what changed between a baseline run and another run (status, targets, recovery, timing). */
export default function Compare({ searchParams }: { searchParams: Promise<{ base?: string; other?: string }> }) {
  const { base, other } = use(searchParams);
  const [data, setData] = useState<Comparison | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!base || !other) { setError("Choose two executions: /compare?base=<execution>&other=<execution>"); return; }
    api<Comparison>(`/executions/${base}/compare/${other}`).then(setData).catch((e) => setError(String(e)));
  }, [base, other]);
  if (error) return <p className="muted">{error}</p>;
  if (!data) return <p className="muted">Comparing…</p>;
  return (
    <>
      <section className="card">
        <h2 style={{ marginTop: 0 }}>Execution comparison</h2>
        <div className="grid2">
          <div><div className="muted">Baseline</div><a href={`/executions/${data.base.id}`}>{data.base.id}</a> <StatusBadge status={data.base.outcome ?? null} />
            <div className="muted">v{data.base.test_case_version} · {data.base.runner}</div></div>
          <div><div className="muted">Compared</div><a href={`/executions/${data.other.id}`}>{data.other.id}</a> <StatusBadge status={data.other.outcome ?? null} />
            <div className="muted">v{data.other.test_case_version} · {data.other.runner}</div></div>
        </div>
        <ul>{data.summary.map((s) => <li key={s}>{s}</li>)}</ul>
        {Object.keys(data.metric_deltas).length > 0 && (
          <table><thead><tr><th>Metric</th><th>Baseline</th><th>Compared</th></tr></thead><tbody>
            {Object.entries(data.metric_deltas).map(([k, v]) => <tr key={k}><td>{k.replaceAll("_", " ")}</td><td>{String(v.base ?? "–")}</td><td>{String(v.other ?? "–")}</td></tr>)}
          </tbody></table>
        )}
      </section>
      <section className="card">
        <table><thead><tr><th>#</th><th>Step</th><th>Baseline</th><th>Compared</th><th>Changes</th></tr></thead><tbody>
          {data.steps.map((r, i) => (
            <tr key={i} style={r.changes.some((c) => c.startsWith("regression")) ? { background: "rgba(211,51,51,.08)" } : undefined}>
              <td>{r.index}</td><td>{r.text}</td><Cell step={r.base} /><Cell step={r.other} />
              <td>{r.changes.length ? r.changes.map((c) => <div key={c}>{c}</div>) : <span className="muted">same</span>}</td>
            </tr>
          ))}
        </tbody></table>
      </section>
    </>
  );
}
