import { useEffect, useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import { useStore } from "@/lib/store";

export default function Dashboard() {
  const { history } = useStore();
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    async function fetchStats() {
      try {
        const res = await fetch("/api/admin/stats");
        setStats(await res.json());
      } catch { /* ignore */ }
    }
    fetchStats();
  }, []);

  const riskTrend = history.slice(0, 10).reverse().map((r, i) => ({
    n: history.length - i,
    risk: Math.round(r.risk_score * 100),
    claims: r.summary.total_claims,
  }));

  const verdictData = history.length > 0 ? [
    { name: "Supported", value: history[0].summary.supported, fill: "#4caf50" },
    { name: "Contradicted", value: history[0].summary.contradicted, fill: "#f44336" },
    { name: "Unverifiable", value: history[0].summary.unverifiable, fill: "#ffc107" },
  ] : [];

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Dashboard</h1>

      {stats && (
        <div className="grid grid-cols-4 gap-4">
          <div className="stat-card"><div className="number">{stats.total_verifications}</div><div className="label">Total</div></div>
          <div className="stat-card risk"><div className="number">{(stats.average_risk_score * 100).toFixed(1)}%</div><div className="label">Avg Risk</div></div>
          <div className="stat-card supported"><div className="number">{stats.total_claims}</div><div className="label">Claims</div></div>
          <div className="stat-card unverifiable"><div className="number">{stats.cache_hit_rate.toFixed(1)}%</div><div className="label">Cache Hit</div></div>
        </div>
      )}

      {riskTrend.length > 0 && (
        <>
          <h2 className="text-lg font-semibold">Risk Trend</h2>
          <LineChart width={600} height={200} data={riskTrend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="n" label="Request" />
            <YAxis label="Risk %" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="risk" stroke="#f44336" name="Risk Score" />
          </LineChart>
        </>
      )}

      {verdictData.length > 0 && (
        <>
          <h2 className="text-lg font-semibold">Verdict Distribution</h2>
          <BarChart width={400} height={200} data={verdictData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        </>
      )}

      {history.length === 0 && (
        <p className="text-gray-400">No verification history yet.</p>
      )}
    </div>
  );
}