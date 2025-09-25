import React from "react";

const OBJECTIVE_LABELS = {
  max_return: "Maximize Returns",
  min_risk: "Minimize Risk",
  max_sharpe: "Maximize Sharpe",
  esg_focus: "ESG Focus",
};

const formatPercent = (value, fractionDigits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  return `${numeric.toFixed(fractionDigits)}%`;
};

export default function PortfolioSummary({ summary }) {
  if (!summary) {
    return null;
  }

  const {
    expected_return,
    volatility,
    sharpe_ratio,
    max_drawdown,
    strategy,
    time_horizon,
    risk_tolerance,
    data_source,
    symbols,
    optimization_objectives,
    risk_metrics,
  } = summary;

  const perfMetrics = [
    {
      label: "Expected Return",
      value: formatPercent(expected_return * 100),
      tone: expected_return >= 0 ? "positive" : "negative",
    },
    {
      label: "Volatility",
      value: formatPercent(volatility * 100),
      tone: volatility <= 0.15 ? "positive" : "neutral",
    },
    {
      label: "Sharpe Ratio",
      value: Number.isFinite(sharpe_ratio) ? sharpe_ratio.toFixed(2) : "—",
      tone: sharpe_ratio >= 1 ? "positive" : "neutral",
    },
    {
      label: "Max Drawdown",
      value: formatPercent(max_drawdown * 100),
      tone: max_drawdown > -0.2 ? "positive" : "negative",
    },
  ];

  const riskRows = risk_metrics
    ? [
        { label: "95% VaR", value: formatPercent(risk_metrics.var_95_pct) },
        { label: "99% VaR", value: formatPercent(risk_metrics.var_99_pct) },
        { label: "95% CVaR", value: formatPercent(risk_metrics.cvar_95_pct) },
        { label: "99% CVaR", value: formatPercent(risk_metrics.cvar_99_pct) },
      ]
    : [];

  const activeObjectives = Object.entries(optimization_objectives || {})
    .filter(([key, value]) => key in OBJECTIVE_LABELS && value)
    .map(([key]) => OBJECTIVE_LABELS[key]);

  return (
    <div className="card summary-card">
      <div className="summary-header">
        <div>
          <h3>Strategy Snapshot</h3>
          <p className="summary-subtitle">
            {strategy || "Custom Strategy"} · {time_horizon}
          </p>
        </div>
        <div className="summary-badge">Risk Tolerance: {risk_tolerance}</div>
      </div>

      <div className="summary-grid">
        {perfMetrics.map((metric) => (
          <div
            key={metric.label}
            className={`summary-card-item ${metric.tone || "neutral"}`}
          >
            <span className="metric-label">{metric.label}</span>
            <span className="metric-value">{metric.value}</span>
          </div>
        ))}
      </div>

      <div className="summary-meta">
        <div>
          <span className="metric-label">Data Source</span>
          <span className="metric-value">{data_source}</span>
        </div>
        <div>
          <span className="metric-label">Tracked Assets</span>
          <span className="metric-value">{symbols?.length || 0}</span>
        </div>
        {activeObjectives.length > 0 && (
          <div className="objective-pills">
            {activeObjectives.map((label) => (
              <span key={label} className="pill">
                {label}
              </span>
            ))}
          </div>
        )}
      </div>

      {riskRows.length > 0 && (
        <div className="risk-grid">
          {riskRows.map((metric) => (
            <div key={metric.label} className="summary-card-item neutral">
              <span className="metric-label">{metric.label}</span>
              <span className="metric-value">{metric.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
