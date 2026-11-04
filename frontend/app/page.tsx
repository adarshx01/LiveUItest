"use client";
import { useEffect, useState } from "react";
import { api } from "@/services/api";
import type { Execution, TestCaseSummary } from "@/types/api";
import { StatusBadge } from "@/components/StatusBadge";

export default function Dashboard() {
  const [tests, setTests] = useState<TestCaseSummary[]>([]);
  const [runs, setRuns] = useState<Execution[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const load = () => {
      api<TestCaseSummary[]>("/test-cases").then(setTests).catch((e) => setError(String(e)));
      api<Execution[]>("/executions?limit=25").then(setRuns).catch(() => undefined);
    };
    load();
    const timer = setInterval(load, 5000);
    return () => clearInterval(timer);
  }, []);
  return (
    <>
      {error && <div className="card" style={{ color: "#c62828" }}>API unavailable: {error}</div>}
      <div className="grid2">
        <section className="card">
          <div className="row" style={{ justifyContent: "space-between" }}><h2 style={{ margin: 0 }}>Test cases</h2><a className="btn" href="/tests/new">New test</a></div>
          <table><thead><tr><th>Name</th><th>Version</th><th>Memory</th><th>Updated</th></tr></thead><tbody>
            {tests.map((t) => (
              <tr key={t.id}><td><a href={`/tests/${t.id}`}>{t.name}</a><div className="muted">{t.id}</div></td>
                <td>v{t.current_version}</td><td><a href={`/memory/${t.application_id}`}>inspect</a></td>
                <td className="muted">{new Date(t.updated_at).toLocaleString()}</td></tr>
            ))}
          </tbody></table>
        </section>
        <section className="card">
          <h2 style={{ marginTop: 0 }}>Recent executions</h2>
          <table><thead><tr><th>Execution</th><th>Status</th><th>Outcome</th><th>Steps</th></tr></thead><tbody>
            {runs.map((r) => (
              <tr key={r.id}><td><a href={`/executions/${r.id}`}>{r.id}</a><div className="muted">{r.test_case_id} v{r.test_case_version}</div></td>
                <td><StatusBadge status={r.status} /></td><td><StatusBadge status={r.outcome} /></td>
                <td className="muted">{r.metrics?.passed_steps ?? "-"}/{r.metrics?.total_steps ?? "-"}{r.metrics?.recovered_steps ? ` · ${r.metrics.recovered_steps} recovered` : ""}</td></tr>
            ))}
          </tbody></table>
        </section>
      </div>
    </>
  );
}
