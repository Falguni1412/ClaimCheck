import { useState } from "react";

/**
 * Hook that loads demo data into the form fields.
 * Clicking the "Load Demo" button or pressing Ctrl+Cmd+Enter triggers this.
 */
export function loadDemo() {
  const [demoLoaded, setDemoLoaded] = useState(false);

  const load = () => {
    setDemoLoaded(true);
  };

  // Demo data
  const demoAnswer = `Metformin should be taken on an empty stomach. It can be combined with insulin therapy. Lactic acidosis occurs in 10% of patients. Nausea is a common side effect.`;

  const demoSources = `Metformin is a medication for type 2 diabetes. Take metformin with meals to reduce stomach upset. Do not take on an empty stomach.

Common side effects of metformin include nausea, diarrhea, and stomach pain. Lactic acidosis is rare, occurring in approximately 1 in 30,000 patient-years.

Metformin can be safely combined with insulin therapy under medical supervision. This combination is commonly prescribed for patients with poorly controlled diabetes.`;

  if (demoLoaded) {
    // In a real implementation, this would set form state
    // For now, just return the demo data
    setDemoLoaded(false);
    return {
      answer: demoAnswer,
      sources: demoSources,
    };
  }

  return { load };
}