"use client";

import { useState } from "react";
import { useStore } from "@/lib/store";

export default function Compare() {
  const [leftAnswer, setLeftAnswer] = useState("");
  const [rightAnswer, setRightAnswer] = useState("");
  const { answer, setAnswer, sources, setSources } = useStore();

  const handleCompare = () => {
    // Merge sources for comparison
    setSources([...sources]);
    setAnswer(leftAnswer);
  };

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Compare Mode</h1>
      <p className="text-gray-400">
        Side-by-side verification of two LLM answers against the same sources.
      </p>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <h2 className="text-sm font-semibold mb-2">Answer A</h2>
          <textarea
            value={leftAnswer}
            onChange={(e) => setLeftAnswer(e.target.value)}
            rows={8}
            className="w-full bg-gray-900 rounded-lg border border-gray-700 p-4 focus:ring-2 focus:ring-purple-500 focus:outline-none resize-none"
          />
        </div>
        <div>
          <h2 className="text-sm font-semibold mb-2">Answer B</h2>
          <textarea
            value={rightAnswer}
            onChange={(e) => setRightAnswer(e.target.value)}
            rows={8}
            className="w-full bg-gray-900 rounded-lg border border-gray-700 p-4 focus:ring-2 focus:ring-purple-500 focus:outline-none resize-none"
          />
        </div>
      </div>

      <button onClick={handleCompare} className="px-6 py-3 bg-purple-600 rounded-md font-medium hover:bg-purple-700">
        Compare
      </button>
    </div>
  );
}