import "./globals.css";
import type { ReactNode } from "react";

export const metadata = { title: "AI E2E Testing Platform", description: "Agentic natural-language browser testing" };

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="app">
          <strong>AI Testing Platform</strong>
          <a href="/">Dashboard</a>
          <a href="/tests/new">New test</a>
          <a href="/approvals">Approvals</a>
        </header>
        <main>{children}</main>
      </body>
    </html>
  );
}
