import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ClaimCheck — Claim-Level LLM Verification Engine",
  description: "Verify LLM claims against ground truth sources with evidence attribution and numerical overrides.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans antialiased bg-slate-950 text-slate-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}