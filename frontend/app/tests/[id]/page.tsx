"use client";
import { use, useEffect, useState } from "react";
import { api } from "@/services/api";
import type { Execution, TestCaseDetail } from "@/types/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function TestDetail({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [tc, setTc] = useState<TestCaseDetail | null>(null);
  const [runs, setRuns] = useState<Execution[]>([]);
  useEffect(() => {
    api<TestCaseDetail>(`/test-cases/${id}`).then(setTc);
    api<Execution[]>(`/executions?test_case_id=${id}`).then(setRuns);
  }, [id]);
  const run = async () => {
    const exec = await api<Execution>("/executions", { method: "POST", body: JSON.stringify({ test_case_id: id }) });
    window.location.href = `/executions/${exec.id}`;
  };
  if (!tc) return <p className="muted">Loading…</p>;
  const outcomes = runs.map((r) => r.outcome).filter(Boolean);
  const flaky = new Set(outcomes).size > 1;
  return (
    <div className="grid2">
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}><h2 style={{ margin: 0 }}>{tc.name}</h2><button onClick={run}>Run</button></div>
        <p className="muted">{tc.id} · v{tc.version} · {tc.source_format} · <a href={`/memory/${tc.application_id}`}>application memory</a></p>
        <pre>{tc.source_text}</pre>
        <table><tbody>{tc.steps.map((s) => <tr key={s.id}><td>{s.index}</td><td>{s.text}</td><td><code>{s.summary}</code></td></tr>)}</tbody></table>
      </section>
      <section className="card">
        <h2 style={{ marginTop: 0 }}>History {flaky && <span className="pill">outcomes vary across runs (possible flakiness)</span>}</h2>
        <div className="row" style={{ marginBottom: 8 }}>{runs.slice().reverse().map((r) => <a key={r.id} href={`/executions/${r.id}`} title={r.id}><StatusBadge status={r.outcome ?? r.status} /></a>)}</div>
        <table><thead><tr><th>Execution</th><th>Outcome</th><th>Recovered</th><th>Memory</th><th></th></tr></thead><tbody>
          {runs.map((r, i) => <tr key={r.id}><td><a href={`/executions/${r.id}`}>{r.id}</a>{r.scope && <span className="pill">partial</span>}</td>
            <td><StatusBadge status={r.outcome ?? r.status} /></td>
            <td>{r.metrics?.recovered_steps ?? 0}</td><td>{r.metrics?.memory_hits ?? 0}/{r.metrics?.memory_lookups ?? 0}</td>
            <td>{runs[i + 1] && <a href={`/compare?base=${runs[i + 1].id}&other=${r.id}`}>compare with previous</a>}</td></tr>)}
        </tbody></table>
      </section>
    </div>
  );
}
