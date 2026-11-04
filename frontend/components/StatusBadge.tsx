const COLORS: Record<string, string> = {
  PASSED: "#138a4a", FAILED: "#c62828", BLOCKED: "#b26a00", INCONCLUSIVE: "#5f6b7a", SKIPPED: "#8a94a6",
  RUNNING: "#3355ff", QUEUED: "#8a94a6", COMPLETED: "#18202f", AWAITING_APPROVAL: "#b26a00",
  AWAITING_HUMAN: "#b26a00", CANCELLED: "#5f6b7a", ERROR: "#c62828", PENDING: "#8a94a6",
};

export function StatusBadge({ status }: { status: string | null | undefined }) {
  const value = status ?? "—";
  return <span className="badge" style={{ background: COLORS[value] ?? "#5f6b7a" }}>{value}</span>;
}

export function statusColor(status: string): string {
  return COLORS[status] ?? "#5f6b7a";
}
