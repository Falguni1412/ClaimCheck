import { useState, useEffect, useRef, useCallback } from "react";
import { streamVerify, StreamingResult } from "@/lib/api";
import { useStore } from "@/lib/store";

/**
 * Hook for streaming verification results via SSE.
 * Returns (startStreaming, stopStreaming) where startStreaming takes a
 * verify callback and yields results incrementally.
 */
export function useStreamingVerify() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentClaim, setCurrentClaim] = useState<StreamingResult | null>(null);
  const [completed, setCompleted] = useState({
    total: 0,
    supported: 0,
    contradicted: 0,
    unverifiable: 0,
    risk_score: 0,
  });
  const [error, setError] = useState<string | null>(null);
  const abortCtrl = useRef<AbortController | null>(null);

  const start = useCallback(
    async (verifyCallback: () => Promise<void>) => {
      if (isStreaming) return;
      setIsStreaming(true);
      setError(null);
      setCurrentClaim(null);

      // Reset completed stats
      setCompleted({ total: 0, supported: 0, contradicted: 0, unverifiable: 0, risk_score: 0 });

      // Clear any previous abort controller
      if (abortCtrl.current) {
        abortCtrl.current.abort();
      }
      abortCtrl.current = new AbortController();

      try {
        const generator = streamVerify({
          answer: useStore.getState().answer,
          sources: useStore.getState().sources,
          top_k: 3,
        });

        for await (const result of generator) {
          // Update current claim
          setCurrentClaim(result);
          setCompleted((prev) => ({
            ...prev,
            total: prev.total + 1,
          }));

          // Update verdict counts
          if (result.verdict === "SUPPORTED") {
            setCompleted((prev) => ({ ...prev, supported: prev.supported + 1 }));
          } else if (result.verdict === "CONTRADICTED") {
            setCompleted((prev) => ({ ...prev, contradicted: prev.contradicted + 1 }));
          } else {
            setCompleted((prev) => ({ ...prev, unverifiable: prev.unverifiable + 1 }));
          }
        }

        // After stream completes, run final verification
        await verifyCallback();

      } catch (e: any) {
        console.error("Streaming verify error:", e);
        setError(e?.message || "Streaming verification failed");
      } finally {
        setIsStreaming(false);
        setCurrentClaim(null);
        // Don't auto-clear completed stats
      }
    },
    [isStreaming]
  );

  const stop = useCallback(() => {
    if (abortCtrl.current) {
      abortCtrl.current.abort();
      abortCtrl.current = new AbortController();
    }
    setIsStreaming(false);
    setCurrentClaim(null);
    setError(null);
    // Keep completed stats
  }, []);

  return { start, stop, isStreaming, currentClaim, completed, error };
}