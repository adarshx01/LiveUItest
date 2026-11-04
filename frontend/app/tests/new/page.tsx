"use client";
import { useEffect, useState } from "react";
import { api } from "@/services/api";
import type { Execution, Project, TestCaseDetail } from "@/types/api";

const SAMPLE = `URL: http://127.0.0.1:8800

Test Case: Verify user login

1. Open the login page.
2. Click on the Login button.
3. Enter username as abc.
4. Enter password as {{test_password}}.
5. Click on the Sign In button.
6. Verify that the user is redirected to the dashboard.
7. Verify that the dashboard displays the logged-in user's name.`;

export default function NewTest() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState("");
  const [content, setContent] = useState(SAMPLE);
  const [url, setUrl] = useState("");
  const [variables, setVariables] = useState("");
  const [secretName, setSecretName] = useState("test_password");
  const [secretValue, setSecretValue] = useState("");
  const [viewport, setViewport] = useState("1280x800");
  const [autoApprove, setAutoApprove] = useState(false);
  const [parsed, setParsed] = useState<TestCaseDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Project[]>("/projects").then(async (list) => {
      if (!list.length) list = [await api<Project>("/projects", { method: "POST", body: JSON.stringify({ name: "Default project" }) })];
      setProjects(list);
      setProjectId(list[0].id);
    }).catch((e) => setError(String(e)));
  }, []);

  const save = async () => {
    setBusy(true); setError(null);
    try {
      if (secretName && secretValue) {
        await api("/secrets", { method: "POST", body: JSON.stringify({ name: secretName, value: secretValue, project_id: projectId }) });
      }
      const tc = await api<TestCaseDetail>("/test-cases", { method: "POST", body: JSON.stringify({ project_id: projectId, content, base_url: url || null }) });
      setParsed(tc);
      return tc;
    } catch (e) { setError(String(e)); return null; } finally { setBusy(false); }
  };

  const run = async () => {
    const tc = parsed ?? (await save());
    if (!tc) return;
    const vars = Object.fromEntries(variables.split("\n").filter((l) => l.includes("=")).map((l) => [l.split("=")[0].trim(), l.split("=").slice(1).join("=").trim()]));
    const [w, h] = viewport.split("x").map(Number);
    const exec = await api<Execution>("/executions", { method: "POST", body: JSON.stringify({ test_case_id: tc.id, variables: vars, options: { viewport: [w, h], auto_approve: autoApprove } }) });
    window.location.href = `/executions/${exec.id}`;
  };

  return (
    <div className="grid2">
      <section className="card">
        <h2 style={{ marginTop: 0 }}>Natural-language test case</h2>
        <label>Project</label>
        <select value={projectId} onChange={(e) => setProjectId(e.target.value)}>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
        <label>Application URL (optional when the test contains a URL line)</label>
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://app.example.com" />
        <label>Test case (plain text, Markdown, JSON or CSV)</label>
        <textarea rows={16} value={content} onChange={(e) => { setContent(e.target.value); setParsed(null); }} />
        <div className="grid2">
          <div><label>Test data (name=value per line)</label><textarea rows={3} value={variables} onChange={(e) => setVariables(e.target.value)} placeholder="username=abc" /></div>
          <div><label>Secret (stored encrypted, never shown again)</label>
            <input value={secretName} onChange={(e) => setSecretName(e.target.value)} placeholder="name" />
            <input type="password" style={{ marginTop: 6 }} value={secretValue} onChange={(e) => setSecretValue(e.target.value)} placeholder="value" /></div>
        </div>
        <div className="grid2">
          <div><label>Viewport</label><select value={viewport} onChange={(e) => setViewport(e.target.value)}>
            <option>1280x800</option><option>1440x900</option><option>390x844</option><option>768x1024</option></select></div>
          <div><label>Environment policy</label><label style={{ color: "#18202f" }}><input type="checkbox" style={{ width: "auto" }} checked={autoApprove} onChange={() => setAutoApprove(!autoApprove)} /> auto-approve well-grounded high-risk actions (test environments only)</label></div>
        </div>
        <div className="row" style={{ marginTop: 12 }}>
          <button className="secondary" disabled={busy} onClick={save}>Analyze</button>
          <button disabled={busy || !projectId} onClick={run}>Save &amp; run</button>
        </div>
        {error && <p style={{ color: "#c62828" }}>{error}</p>}
      </section>
      <section className="card">
        <h2 style={{ marginTop: 0 }}>Structured plan</h2>
        {!parsed && <p className="muted">Analyze the test to see how each instruction was understood.</p>}
        {parsed && (
          <>
            <p className="muted">{parsed.id} · v{parsed.version}</p>
            {parsed.issues.length > 0 && parsed.issues.map((i, k) => <div key={k} className="pill">{i.severity}: {i.reason}</div>)}
            <table><thead><tr><th>#</th><th>Instruction</th><th>Understood as</th><th>Conf.</th></tr></thead><tbody>
              {parsed.steps.map((s) => <tr key={s.id}><td>{s.index}</td><td>{s.text}</td><td><code>{s.summary}</code></td><td>{s.confidence.toFixed(2)}</td></tr>)}
            </tbody></table>
          </>
        )}
      </section>
    </div>
  );
}
