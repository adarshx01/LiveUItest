"use client";
import { useEffect, useRef, useState } from "react";
import { api } from "@/services/api";
import type { Observation } from "@/types/api";

/** Live screenshot with detected-element overlay; when the run awaits a human, clicks are forwarded (audited). */
export function LiveBrowser({ executionId, active, awaitingHuman }: { executionId: string; active: boolean; awaitingHuman: boolean }) {
  const [image, setImage] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [observation, setObservation] = useState<Observation | null>(null);
  const [overlay, setOverlay] = useState(true);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (!active) return;
    let stop = false;
    const tick = async () => {
      try {
        const shot = await api<{ url: string; image: string }>(`/executions/${executionId}/live`);
        if (!stop) { setImage(shot.image); setUrl(shot.url); }
        const obs = await api<Observation>(`/executions/${executionId}/observation`);
        if (!stop) setObservation(obs);
      } catch { /* browser busy or finished */ }
      if (!stop) setTimeout(tick, 2500);
    };
    tick();
    return () => { stop = true; };
  }, [executionId, active]);

  const onClick = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (!awaitingHuman || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const scale = imgRef.current.naturalWidth / rect.width;
    await api(`/executions/${executionId}/browser/human-action`, {
      method: "POST", body: JSON.stringify({ type: "click", x: (e.clientX - rect.left) * scale, y: (e.clientY - rect.top) * scale }),
    });
  };

  if (!active) return <div className="muted">The live browser is shown while the execution runs.</div>;
  if (!image) return <div className="muted">Waiting for the browser…</div>;
  const scale = imgRef.current ? imgRef.current.getBoundingClientRect().width / (imgRef.current.naturalWidth || 1) : 1;
  return (
    <div>
      <div className="row muted"><span>{url}</span><label style={{ margin: 0 }}><input type="checkbox" style={{ width: "auto" }} checked={overlay} onChange={() => setOverlay(!overlay)} /> element overlay</label>
        {awaitingHuman && <strong style={{ color: "#b26a00" }}>Human verification required — click in the screenshot to operate the browser</strong>}</div>
      <div className="shot">
        <img ref={imgRef} src={image} alt="live browser" onClick={onClick} style={{ cursor: awaitingHuman ? "crosshair" : "default" }} />
        {overlay && observation?.elements.filter((el) => el.bbox && el.visible && el.bbox.width > 0).slice(0, 120).map((el) => (
          <div key={el.element_id} className="box dim" title={`${el.role} "${el.name}" ${el.purpose ?? ""}`}
            style={{ left: el.bbox!.x * scale, top: el.bbox!.y * scale, width: el.bbox!.width * scale, height: el.bbox!.height * scale }} />
        ))}
      </div>
    </div>
  );
}
