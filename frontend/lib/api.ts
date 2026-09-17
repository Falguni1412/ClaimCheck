/**
 * API client for ClaimCheck backend.
 * Provides typed request/response helpers for the frontend.
 */
import { useStore } from "./store";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ── request helpers ──────────────────────────────────────────────────
function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  return fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(useStore.getState().authToken
        ? { Authorization: `Bearer ${useStore.getState().authToken}` }
        : {}),
    },
    ...init,
  }).then(async (res) => {
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed: ${res.status}`);
    }
    return res.json() as Promise<T>;
  });
}

// ── typed endpoints ──────────────────────────────────────────────────

export interface VerifyRequest {
  answer: string;
  sources: string[];
  use_chunking?: boolean;
  top_k?: number;
}

export interface ClaimResult {
  claim: string;
  verdict: "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE";
  confidence: number;
  evidence: string;
  verdict_scores?: {
    supported: number;
    unverifiable: number;
    contradicted: number;
  };
  numerical_check?: {
    consistent: boolean;
    claim_value: number;
    evidence_value: number;
    ratio: number;
    note: string;
    unit_mismatch?: string;
  };
  explanation?: string;
}

export interface VerifyResponse {
  request_id: string;
  claims: ClaimResult[];
  risk_score: number;
  summary: {
    total_claims: number;
    supported: number;
    contradicted: number;
    unverifiable: number;
    processing_time_seconds: number;
    average_confidence?: number;
  };
  metadata: {
    model: string;
    sources_processed: number;
    ensemble: boolean;
    timestamp: number;
    cached?: boolean;
  };
  cached?: boolean;
}

export async function verifyClaims(
  data: VerifyRequest
): Promise<VerifyResponse> {
  return request<VerifyResponse>("/verify", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export interface BatchVerifyRequest {
  items: VerifyRequest[];
  parallel?: boolean;
  fail_fast?: boolean;
}

export interface BatchVerifyResponse {
  results: VerifyResponse[];
  total: number;
  successful: number;
  failed: number;
  total_processing_time_seconds: number;
}

export async function verifyBatch(
  data: BatchVerifyRequest
): Promise<BatchVerifyResponse> {
  return request<BatchVerifyResponse>("/verify/batch", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export interface StreamingResult {
  claim: string;
  verdict: "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE";
  confidence: number;
  evidence: string;
  verdict_scores?: {
    supported: number;
    unverifiable: number;
    contradicted: number;
  };
  numerical_check?: any;
  explanation?: string;
}

export interface StreamVerificationResponse {
  request_id: string;
  total_claims: number;
}

export async function streamVerify(
  data: VerifyRequest
): Promise<AsyncGenerator<StreamingResult>> {
  const url = `${API_BASE}/verify/stream`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 min timeout

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(useStore.getState().authToken
          ? { Authorization: `Bearer ${useStore.getState().authToken}` }
          : {}),
      },
      body: JSON.stringify(data),
      signal: controller.signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Stream failed: ${response.status}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // SSE messages are newline-delimited
      while (buffer.includes("data: ")) {
        const idx = buffer.indexOf("data: ");
        const message = buffer.substring(idx + 6);
        buffer = buffer.substring(0, idx);

        // Skip empty lines / comments
        if (!message.trim() || message.startsWith(":")) continue;

        try {
          const parsed = JSON.parse(message) as StreamingResult;
          if (parsed.claim) {
            yield parsed;
          }
        } catch (e) {
          console.warn("Failed to parse SSE message:", e, "raw:", message);
        }
      }
    }
  } catch (e: any) {
    if (e.name !== "AbortError") {
      console.error("Stream verification error:", e);
    }
    // Yield error result
    yield {
      claim: "",
      verdict: "UNVERIFIABLE",
      confidence: 0,
      evidence: "",
      explanation: e?.message || "Stream verification failed",
    } as StreamingResult;
  } finally {
    clearTimeout(timeoutId);
  }
}

export { streamVerify };