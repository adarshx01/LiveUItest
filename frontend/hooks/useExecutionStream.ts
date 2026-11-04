"use client";
import { useEffect, useRef, useState } from "react";
import { api, streamUrl } from "@/services/api";
import type { ExecutionEvent } from "@/types/api";

/** Replays persisted events, then follows the live WebSocket stream. */
export function useExecutionStream(executionId: string) {
  const [events, setEvents] = useState<ExecutionEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const seen = useRef(new Set<number>());

  useEffect(() => {
    let socket: WebSocket | null = null;
    let cancelled = false;
    const add = (items: ExecutionEvent[]) => {
      const fresh = items.filter((e) => e.seq && !seen.current.has(e.seq));
      fresh.forEach((e) => seen.current.add(e.seq));
      if (fresh.length) setEvents((prev) => [...prev, ...fresh].sort((a, b) => a.seq - b.seq));
    };
    api<ExecutionEvent[]>(`/executions/${executionId}/events`).then(add).catch(() => undefined);
    try {
      socket = new WebSocket(streamUrl(executionId));
      socket.onopen = () => !cancelled && setConnected(true);
      socket.onclose = () => !cancelled && setConnected(false);
      socket.onmessage = (message) => {
        const data = JSON.parse(message.data) as ExecutionEvent;
        if (data.event !== "PING") add([data]);
      };
    } catch {
      setConnected(false);
    }
    return () => {
      cancelled = true;
      socket?.close();
    };
  }, [executionId]);

  return { events, connected };
}
