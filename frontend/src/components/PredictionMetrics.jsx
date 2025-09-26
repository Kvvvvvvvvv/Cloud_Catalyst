import React from "react";

const formatMetric = (value, unit = "", decimals = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  return `${numeric.toFixed(decimals)}${unit}`;
};

const getMetricTone = (metric, value) => {
  switch (metric) {
    case "mae":
      return value < 5 ? "positive" : value < 15 ? "neutral" : "negative";
    case "rmse":
      return value < 5 ? "positive" : value < 15 ? "neutral" : "negative";
    case "mape":
      return value < 5 ? "positive" : value < 10 ? "neutral" : "negative";
    case "r_squared":
      return value > 0.8 ? "positive" : value > 0.6 ? "neutral" : "negative";
    case "sharpe_ratio":
      return value > 2 ? "positive" : value > 1 ? "neutral" : "negative";
    case "directional_accuracy":
      return value > 65 ? "positive" : value > 55 ? "neutral" : "negative";
    default:
      return "neutral";
  }
};

const METRIC_DESCRIPTIONS = {
  mae: {
    name: "Mean Absolute Error (MAE)",
    description: "Average magnitude of prediction errors. Lower = better accuracy.",
    unit: "",
    tooltip: "Shows how far off predictions are on average. A MAE of $5 means predictions are typically $5 away from actual prices."
  },
  rmse: {
    name: "Root Mean Squared Error (RMSE)", 
    description: "Penalizes larger errors more heavily. Lower = better fit.",
    unit: "",
    tooltip: "Emphasizes larger prediction errors. If RMSE >> MAE, the model struggles with extreme price movements."
  },
  mape: {
    name: "Mean Absolute Percentage Error (MAPE)",
    description: "Error as percentage of actual price. Useful for trading decisions.",
    unit: "%",
    tooltip: "Shows prediction accuracy as a percentage. 5% MAPE means predictions are typically 5% off from actual prices."
  },
  r_squared: {
    name: "R² (Coefficient of Determination)",
    description: "How much variance in prices is captured. Higher = better model.",
    unit: "",
    tooltip: "Shows how well the model explains price movements. 0.8 means 80% of price variance is explained by the model."
  },
  sharpe_ratio: {
    name: "Sharpe Ratio (Financial Metric)",
    description: "Risk-adjusted return from trading strategy. Higher = better.",
    unit: "",
    tooltip: "Shows if prediction-based trading is profitable after accounting for risk. > 1 is good, > 2 is excellent."
  },
  directional_accuracy: {
    name: "Directional Accuracy",
    description: "Percentage of correct price direction predictions.",
    unit: "%",
    tooltip: "Critical for trading strategies. 60% means the model correctly predicts price direction 6 out of 10 times."
  },
  volatility_explained: {
    name: "Volatility Explained",
    description: "How well model explains volatility patterns.",
    unit: "",
    tooltip: "Shows how accurately the model captures market volatility characteristics."
  },
  prediction_confidence: {
    name: "Prediction Confidence",
    description: "Model consistency and reliability score.",
    unit: "",
    tooltip: "Higher values indicate more consistent and reliable predictions."
  }
};

export default function PredictionMetrics({ predictionMetrics }) {
  if (!predictionMetrics) {
    return (
      <div className="card">
        <h3>Prediction Accuracy Metrics</h3>
        <div className="empty-card">
          <p>Run a portfolio simulation to see prediction accuracy metrics</p>
        </div>
      </div>
    );
  }

  const { portfolio_metrics, individual_metrics, portfolio_interpretation, recommendation } = predictionMetrics;

  if (!portfolio_metrics) {
    return (
      <div className="card">
        <h3>Prediction Accuracy Metrics</h3>
        <div className="empty-card">
          <p>No prediction metrics available</p>
        </div>
      </div>
    );
  }

  const coreMetrics = [
    {
      key: "weighted_mae",
      label: "MAE (Mean Absolute Error)",
      value: portfolio_metrics.weighted_mae,
      unit: "",
      tone: getMetricTone("mae", portfolio_metrics.weighted_mae)
    },
    {
      key: "weighted_rmse", 
      label: "RMSE (Root Mean Squared Error)",
      value: portfolio_metrics.weighted_rmse,
      unit: "",
      tone: getMetricTone("rmse", portfolio_metrics.weighted_rmse)
    },
    {
      key: "weighted_mape",
      label: "MAPE (Mean Absolute Percentage Error)",
      value: portfolio_metrics.weighted_mape,
      unit: "%",
      tone: getMetricTone("mape", portfolio_metrics.weighted_mape)
    },
    {
      key: "weighted_r2",
      label: "R² (Coefficient of Determination)",
      value: portfolio_metrics.weighted_r2,
      unit: "",
      tone: getMetricTone("r_squared", portfolio_metrics.weighted_r2)
    },
    {
      key: "weighted_sharpe",
      label: "Sharpe Ratio (Financial Metric)",
      value: portfolio_metrics.weighted_sharpe,
      unit: "",
      tone: getMetricTone("sharpe_ratio", portfolio_metrics.weighted_sharpe)
    }
  ];

  const getRecommendationTone = (rec) => {
    if (rec.includes("STRONG BUY")) return "positive";
    if (rec.includes("BUY")) return "positive";
    if (rec.includes("HOLD")) return "neutral";
    if (rec.includes("AVOID")) return "negative";
    return "neutral";
  };

  return (
    <div className="card">
      <div className="summary-header">
        <div>
          <h3>📊 Prediction Accuracy Metrics</h3>
          <p className="summary-subtitle">
            Portfolio-wide prediction performance analysis
          </p>
        </div>
        <div className={`summary-badge ${getRecommendationTone(recommendation)}`}>
          {recommendation}
        </div>
      </div>

      {/* Core Metrics Grid */}
      <div className="summary-grid">
        {coreMetrics.map((metric) => (
          <div
            key={metric.key}
            className={`summary-card-item ${metric.tone}`}
            title={METRIC_DESCRIPTIONS[metric.key.replace('weighted_', '')]?.tooltip}
          >
            <span className="metric-label">{metric.label}</span>
            <span className="metric-value">
              {formatMetric(metric.value, metric.unit)}
            </span>
          </div>
        ))}
      </div>

      {/* Interpretations */}
      {portfolio_interpretation && (
        <div className="metrics-interpretation">
          <h4>📈 Performance Analysis</h4>
          <div className="interpretation-grid">
            {Object.entries(portfolio_interpretation).map(([key, interpretation]) => (
              <div key={key} className="interpretation-item">
                <div className="metric-name">
                  {METRIC_DESCRIPTIONS[key]?.name || key.toUpperCase()}
                </div>
                <div className="metric-interpretation">{interpretation}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Portfolio Summary Stats */}
      <div className="summary-meta">
        <div>
          <span className="metric-label">Symbols Analyzed</span>
          <span className="metric-value">
            {portfolio_metrics.symbols_analyzed} / {portfolio_metrics.total_symbols}
          </span>
        </div>
        <div>
          <span className="metric-label">Analysis Quality</span>
          <span className="metric-value">
            {portfolio_metrics.weighted_r2 > 0.8 ? "Excellent" :
             portfolio_metrics.weighted_r2 > 0.6 ? "Good" :
             portfolio_metrics.weighted_r2 > 0.4 ? "Moderate" : "Poor"}
          </span>
        </div>
        <div>
          <span className="metric-label">Trading Suitability</span>
          <span className="metric-value">
            {portfolio_metrics.weighted_mape < 5 ? "Excellent" :
             portfolio_metrics.weighted_mape < 10 ? "Good" :
             portfolio_metrics.weighted_mape < 15 ? "Moderate" : "Poor"}
          </span>
        </div>
      </div>

      {/* Individual Stock Metrics Summary */}
      {individual_metrics && individual_metrics.length > 0 && (
        <div className="individual-metrics-summary">
          <h4>🎯 Individual Stock Performance</h4>
          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>MAE</th>
                  <th>MAPE (%)</th>
                  <th>R²</th>
                  <th>Sharpe</th>
                  <th>Quality</th>
                </tr>
              </thead>
              <tbody>
                {individual_metrics.map((stockMetric) => (
                  <tr key={stockMetric.symbol}>
                    <td className="font-bold">{stockMetric.symbol}</td>
                    <td>{formatMetric(stockMetric.metrics.mae)}</td>
                    <td>{formatMetric(stockMetric.metrics.mape, "%")}</td>
                    <td>{formatMetric(stockMetric.metrics.r_squared)}</td>
                    <td>{formatMetric(stockMetric.metrics.sharpe_ratio)}</td>
                    <td>
                      <span className={`pill ${getMetricTone("r_squared", stockMetric.metrics.r_squared)}`}>
                        {stockMetric.metrics.r_squared > 0.8 ? "Excellent" :
                         stockMetric.metrics.r_squared > 0.6 ? "Good" :
                         stockMetric.metrics.r_squared > 0.4 ? "Moderate" : "Poor"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Metric Explanations */}
      <div className="metrics-help">
        <h4>📚 Understanding the Metrics</h4>
        <div className="help-grid">
          <div className="help-item">
            <strong>For Trading Decisions:</strong>
            <span>Focus on MAPE &lt; 10%, Sharpe Ratio &gt; 1, and high Directional Accuracy</span>
          </div>
          <div className="help-item">
            <strong>For Model Quality:</strong>
            <span>Look for R² &gt; 0.6 and RMSE close to MAE for consistent predictions</span>
          </div>
          <div className="help-item">
            <strong>For Risk Management:</strong>
            <span>Monitor Sharpe Ratio and ensure RMSE doesn't significantly exceed MAE</span>
          </div>
        </div>
      </div>
    </div>
  );
}