export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? "dev-api-key";

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}/api/v1${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", "X-API-Key": API_KEY, ...(init.headers ?? {}) },
    cache: "no-store",
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail.slice(0, 300)}`);
  }
  return response.json() as Promise<T>;
}

export function artifactUrl(uri: string | null | undefined): string | null {
  if (!uri || !uri.startsWith("artifact://")) return null;
  const [executionId, kind, name] = uri.slice("artifact://".length).split("/");
  return `${API_URL}/api/v1/artifacts/${executionId}/${kind}/${name}?api_key=${encodeURIComponent(API_KEY)}`;
}

export function streamUrl(executionId: string): string {
  return `${API_URL.replace(/^http/, "ws")}/api/v1/executions/${executionId}/stream?api_key=${encodeURIComponent(API_KEY)}`;
}
