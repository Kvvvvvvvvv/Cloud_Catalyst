import React from "react";

const formatNumber = (value) => {
  if (value === null || value === undefined) {
    return "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  return numeric.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
};

const formatDelta = (value) => {
  if (value === null || value === undefined) {
    return "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  const sign = numeric > 0 ? "+" : "";
  return `${sign}${numeric.toFixed(2)}%`;
};

export default function MarketOverview({ metrics }) {
  const items = (metrics || []).filter((item) => item && !item.error);
  if (!items.length) {
    return null;
  }

  return (
    <div className="card market-card">
      <h3>Market Snapshot</h3>
      <div className="metric-grid">
        {items.map((item) => {
          const deltaClass =
            item.change_pct === null || item.change_pct === undefined
              ? "neutral"
              : item.change_pct >= 0
              ? "positive"
              : "negative";
          return (
            <div key={item.label || item.ticker} className="metric-card">
              <span className="metric-title">{item.label || item.ticker}</span>
              <span className="metric-value">{formatNumber(item.value)}</span>
              <span className={`metric-delta ${deltaClass}`}>
                {formatDelta(item.change_pct)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
