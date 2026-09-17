import { create } from "zustand";

export interface Claim {
  text: string;
  verdict: "SUPPORTED" | "CONTRADICTED" | "UNVERIFIABLE";
  confidence: number;
  evidence: string;
  verdict_scores?: { supported: number; unverifiable: number; contradicted: number };
  numerical_check?: any;
  explanation?: string;
}

export interface VerificationResult {
  request_id: string;
  claims: Claim[];
  risk_score: number;
  summary: {
    total_claims: number;
    supported: number;
    contradicted: number;
    unverifiable: number;
    processing_time_seconds: number;
    average_confidence?: number;
  };
  metadata: { model: string; sources_processed: number; ensemble: boolean; timestamp: number };
  cached?: boolean;
}

interface ClaimCheckStore {
  // Input
  answer: string;
  sources: string[];
  setAnswer: (v: string) => void;
  setSources: (v: string[]) => void;

  // Output
  result: VerificationResult | null;
  setResult: (v: VerificationResult) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
  error: string | null;
  setError: (v: string | null) => void;

  // Auth
  authToken: string | null;
  setAuthToken: (v: string | null) => void;

  // History
  history: VerificationResult[];
  addHistory: (r: VerificationResult) => void;
  clearHistory: () => void;
}

export const useStore = create<ClaimCheckStore>((set) => ({
  answer: "",
  sources: [],
  setAnswer: (v) => set({ answer: v }),
  setSources: (v) => set({ sources: v }),
  result: null,
  setResult: (v) => set({ result: v }),
  loading: false,
  setLoading: (v) => set({ loading: v }),
  error: null,
  setError: (v) => set({ error: v }),
  authToken: null,
  setAuthToken: (v) => set({ authToken: v }),
  history: [],
  addHistory: (r) => set((s) => ({ history: [r, ...s.history].slice(0, 50) })),
  clearHistory: () => set({ history: [] }),
}));