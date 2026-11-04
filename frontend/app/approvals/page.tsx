"use client";
import { useEffect, useState } from "react";
import { api } from "@/services/api";
import type { Approval } from "@/types/api";

export default function Approvals() {
  const [items, setItems] = useState<(Approval & { execution_id: string })[]>([]);
  const load = () => api<(Approval & { execution_id: string })[]>("/approvals").then(setItems);
  useEffect(() => { load(); const t = setInterval(load, 4000); return () => clearInterval(t); }, []);
  const decide = async (id: string, ok: boolean) => { await api(`/approvals/${id}/${ok ? "approve" : "reject"}`, { method: "POST", body: "{}" }); load(); };
  return (
    <section className="card">
      <h2 style={{ marginTop: 0 }}>Pending approvals</h2>
      {!items.length && <p className="muted">Nothing is waiting for a decision.</p>}
      <table><tbody>{items.map((a) => (
        <tr key={a.id}><td><a href={`/executions/${a.execution_id}`}>{a.execution_id}</a><div className="muted">{a.kind} · {a.step_id}</div></td>
          <td><pre>{JSON.stringify(a.request, null, 2)}</pre></td>
          <td>{a.kind === "captcha" ? <a className="btn" href={`/executions/${a.execution_id}`}>Open live browser</a>
            : <div className="row"><button onClick={() => decide(a.id, true)}>Approve</button><button className="danger" onClick={() => decide(a.id, false)}>Reject</button></div>}</td></tr>
      ))}</tbody></table>
    </section>
  );
}
