"use client";
import { use, useEffect, useState } from "react";
import { api } from "@/services/api";
import type { LocatorMemory } from "@/types/api";
import { StatusBadge } from "@/components/StatusBadge";

interface MemoryView {
  locators: LocatorMemory[];
  pages: { route: string; kind: string; structure_hash: string; seen: number }[];
  failures: { category: string; action_signature: string; route: string; root_cause: string; successful_alternative: Record<string, unknown> | null }[];
}

export default function MemoryInspector({ params }: { params: Promise<{ appId: string }> }) {
  const { appId } = use(params);
  const [data, setData] = useState<MemoryView | null>(null);
  const load = () => api<MemoryView>(`/memory/applications/${appId}`).then(setData);
  useEffect(() => { load(); }, [appId]);
  const invalidate = async (id: string) => { await api(`/memory/${id}/invalidate?reason=manual`, { method: "POST" }); load(); };
  if (!data) return <p className="muted">Loading…</p>;
  return (
    <>
      <section className="card">
        <h2 style={{ marginTop: 0 }}>Locator memory · {appId}</h2>
        <p className="muted">Semantic strategies only (never snapshot refs, coordinates or secrets). Every strategy is re-grounded on the live page before use; stale versions are superseded, not deleted.</p>
        <table><thead><tr><th>Route</th><th>Action</th><th>Strategy</th><th>Verified</th><th>Conf.</th><th>Version</th><th>Status</th><th></th></tr></thead><tbody>
          {data.locators.map((m) => (
            <tr key={m.memory_id}><td>{m.route_pattern}</td><td><code>{m.action_signature}</code></td>
              <td>{Object.entries(m.strategy).filter(([k]) => ["role", "name", "label", "test_id", "form_context"].includes(k)).map(([k, v]) => <span key={k} className="pill">{k}: {String(v)}</span>)}</td>
              <td>{m.successes} ok / {m.failures} fail · {m.validated_runs} passed runs</td><td>{m.confidence.toFixed(2)}</td><td>v{m.version}</td>
              <td><StatusBadge status={m.status === "active" ? "PASSED" : m.status === "superseded" ? "INCONCLUSIVE" : "FAILED"} /> <span className="muted">{m.status}</span></td>
              <td>{m.status === "active" && <button className="secondary" onClick={() => invalidate(m.memory_id)}>Invalidate</button>}</td></tr>
          ))}
        </tbody></table>
      </section>
      <div className="grid2">
        <section className="card"><h3 style={{ marginTop: 0 }}>Known pages</h3>
          <table><tbody>{data.pages.map((p) => <tr key={p.structure_hash}><td>{p.route}</td><td>{p.kind}</td><td className="muted">{p.structure_hash}</td><td>seen {p.seen}×</td></tr>)}</tbody></table></section>
        <section className="card"><h3 style={{ marginTop: 0 }}>Failure memory</h3>
          <table><tbody>{data.failures.map((f, i) => <tr key={i}><td>{f.category}</td><td><code>{f.action_signature}</code></td><td>{f.root_cause}</td>
            <td className="muted">{f.successful_alternative ? `healed → ${String(f.successful_alternative.name ?? "")}` : "—"}</td></tr>)}</tbody></table></section>
      </div>
    </>
  );
}
