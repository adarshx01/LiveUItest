"use client";
import { use, useEffect, useState } from "react";
import { api, API_KEY, API_URL } from "@/services/api";
import type { Execution, Observation, StepResult } from "@/types/api";
import { StatusBadge } from "@/components/StatusBadge";
import { StepPanel } from "@/components/StepPanel";
import { LiveBrowser } from "@/components/LiveBrowser";
import { useExecutionStream } from "@/hooks/useExecutionStream";

const ACTIVE = new Set(["QUEUED", "RUNNING", "AWAITING_APPROVAL", "AWAITING_HUMAN"]);

export default function ExecutionView({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [run, setRun] = useState<Execution | null>(null);
  const [steps, setSteps] = useState<StepResult[]>([]);
  const [tab, setTab] = useState<"live" | "steps" | "inspect" | "report">("live");
  const [observation, setObservation] = useState<Observation | null>(null);
  const { events, connected } = useExecutionStream(id);

  useEffect(() => {
    let stop = false;
    const load = async () => {
      try {
        const r = await api<Execution>(`/executions/${id}`);
        if (stop) return;
        setRun(r);
        setSteps(await api<StepResult[]>(`/executions/${id}/steps`));
        if (ACTIVE.has(r.status)) setTimeout(load, 2000);
      } catch { if (!stop) setTimeout(load, 3000); }
    };
    load();
    return () => { stop = true; };
  }, [id]);

  useEffect(() => {
    if (tab === "inspect") api<Observation>(`/executions/${id}/observation`).then(setObservation).catch(() => setObservation(null));
  }, [tab, id, events.length]);

  if (!run) return <p className="muted">Loading…</p>;
  const active = ACTIVE.has(run.status);
  const pending = run.approvals?.find((a) => a.status === "pending");
  const current = [...events].reverse().find((e) => e.event === "STEP_STARTED");
  const decide = async (approve: boolean) => {
    if (!pending) return;
    await api(`/approvals/${pending.id}/${approve ? "approve" : "reject"}`, { method: "POST", body: JSON.stringify({ note: "from UI" }) });
  };
  const resume = async () => api(`/executions/${id}/resume`, { method: "POST", body: JSON.stringify({ approved: true }) });

  return (
    <>
      <section className="card">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div><h2 style={{ margin: 0 }}>Execution {id}</h2>
            <div className="muted">test case <a href={`/tests/${run.test_case_id}`}>{run.test_case_id}</a> v{run.test_case_version} · runner {run.runner ?? "…"} · stream {connected ? "connected" : "offline"}</div></div>
          <div className="row"><StatusBadge status={run.status} /><StatusBadge status={run.outcome} />
            {active && <button className="danger" onClick={() => api(`/executions/${id}/cancel`, { method: "POST" })}>Cancel</button>}
            {!active && <button className="secondary" onClick={async () => { const r = await api<Execution>(`/executions/${id}/retry`, { method: "POST" }); window.location.href = `/executions/${r.id}`; }}>Re-run</button>}</div>
        </div>
        {run.scope && <p className="muted">Partial re-run of {run.scope.step_ids.join(", ")}
          {run.scope.with_prerequisites ? " (earlier steps re-run to rebuild page state)" : " (isolated)"}</p>}
        {run.retry_of && <p className="muted">Re-run of <a href={`/executions/${run.retry_of}`}>{run.retry_of}</a> ·{" "}
          <a href={`/compare?base=${run.retry_of}&other=${id}`}>compare with it</a></p>}
        {run.reason && <p>{run.reason}</p>}
        {current && active && <p><strong>Current step:</strong> {current.message}</p>}
        {pending && (
          <div className="card" style={{ background: "#fff8ec", borderColor: "#f1c27d" }}>
            <strong>{pending.kind === "captcha" ? "Human verification required" : "Approval required"}</strong>
            <pre>{JSON.stringify(pending.request, null, 2)}</pre>
            {pending.kind === "captcha"
              ? <div className="row"><span className="muted">Complete the challenge in the live browser below, then</span><button onClick={resume}>Resume</button></div>
              : <div className="row"><button onClick={() => decide(true)}>Approve</button><button className="danger" onClick={() => decide(false)}>Reject</button></div>}
          </div>
        )}
        {run.metrics && Object.keys(run.metrics).length > 0 && (
          <div className="row">{["total_steps", "passed_steps", "failed_steps", "recovered_steps", "assertions_passed", "memory_hits", "llm_calls", "tool_calls"].map((k) =>
            <span key={k} className="pill">{k.replace(/_/g, " ")}: {run.metrics[k] ?? 0}</span>)}</div>
        )}
      </section>
      <div className="tabs">{(["live", "steps", "inspect", "report"] as const).map((t) => <button key={t} className={tab === t ? "on" : ""} onClick={() => setTab(t)}>{t}</button>)}</div>
      {tab === "live" && (
        <div className="grid2">
          <section className="card"><h3 style={{ marginTop: 0 }}>Live browser</h3><LiveBrowser executionId={id} active={active} awaitingHuman={run.status === "AWAITING_HUMAN"} /></section>
          <section className="card"><h3 style={{ marginTop: 0 }}>Agent activity</h3>
            <div className="timeline">{events.filter((e) => e.event !== "PAGE_OBSERVED").map((e) => (
              <div key={e.seq}><code>{e.event}</code> {e.step_id && <span className="muted">{e.step_id}</span>} {e.message}</div>))}</div>
          </section>
        </div>
      )}
      {tab === "steps" && <section className="card">{steps.map((s) => <StepPanel key={s.step_id} step={s}
        onRerun={active ? undefined : async (stepId) => {
          const r = await api<Execution>(`/executions/${id}/rerun-steps`, { method: "POST", body: JSON.stringify({ step_ids: [stepId] }) });
          window.location.href = `/executions/${r.id}`;
        }} />)}</section>}
      {tab === "inspect" && (
        <section className="card">
          {!observation && <p className="muted">Browser inspection is available while the execution is live.</p>}
          {observation && (
            <div className="grid2">
              <div><h3>Page state</h3><pre>{JSON.stringify({ ...observation.summary, ...observation.page_state }, null, 2)}</pre>
                <h3>Forms</h3><pre>{JSON.stringify(observation.forms, null, 2)}</pre></div>
              <div><h3>Detected elements ({observation.elements.length})</h3>
                <table><thead><tr><th>Role</th><th>Name</th><th>Purpose</th><th>Context</th><th>Box</th></tr></thead><tbody>
                  {observation.elements.slice(0, 150).map((e) => <tr key={e.element_id}><td>{e.role}</td><td>{e.name}</td><td>{e.purpose}</td>
                    <td className="muted">{e.form ?? e.dialog ?? ""}</td><td className="muted">{e.bbox ? `${Math.round(e.bbox.x)},${Math.round(e.bbox.y)}` : ""}</td></tr>)}
                </tbody></table>
                <h3>Accessibility snapshot</h3><pre>{observation.snapshot}</pre></div>
            </div>
          )}
        </section>
      )}
      {tab === "report" && (
        <section className="card">
          {run.report
            ? <iframe title="report" src={`${API_URL}${run.report}?api_key=${encodeURIComponent(API_KEY)}`} style={{ width: "100%", height: "80vh", border: 0 }} />
            : <p className="muted">The report is generated when the execution finishes.</p>}
        </section>
      )}
    </>
  );
}
