import React from "react";

const DATA_SOURCES = ["Yahoo Finance", "Alpha Vantage", "Polygon.io"];
const STRATEGIES = [
  "Balanced Growth",
  "Aggressive Growth",
  "Conservative Income",
  "ESG Focused",
  "Custom",
];
const HORIZONS = ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years"];

const OBJECTIVE_LABELS = {
  max_return: "Maximize Returns",
  min_risk: "Minimize Risk",
  max_sharpe: "Maximize Sharpe Ratio",
  esg_focus: "ESG Considerations",
};

export default function ConfigPanel({
  state,
  onChange,
  onObjectivesChange,
  onRun,
  loading,
}) {
  const {
    symbolsInput,
    weightsInput,
    optimize,
    initialInvestment,
    simulationRuns,
    dataSource,
    strategy,
    timeHorizon,
    riskTolerance,
    objectives,
  } = state;

  const symbolCount = symbolsInput
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean).length;

  return (
    <div className="card panel-card">
      <h3 className="panel-title">Configuration</h3>

      <div className="section-heading">Market Inputs</div>
      <div className="field">
        <label>Symbols (comma-separated)</label>
        <input
          className="input"
          value={symbolsInput}
          onChange={(e) => onChange({ symbolsInput: e.target.value })}
          placeholder="e.g. AAPL, MSFT, GOOG"
        />
      </div>
      <div className="field">
        <label>Weights (comma-separated)</label>
        <input
          className="input"
          value={weightsInput}
          onChange={(e) => onChange({ weightsInput: e.target.value })}
          placeholder="Leave blank to auto-optimize"
        />
      </div>

      <div className="field grid-2">
        <div>
          <label>Data Source</label>
          <select
            className="input"
            value={dataSource}
            onChange={(e) => onChange({ dataSource: e.target.value })}
          >
            {DATA_SOURCES.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Time Horizon</label>
          <select
            className="input"
            value={timeHorizon}
            onChange={(e) => onChange({ timeHorizon: e.target.value })}
          >
            {HORIZONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="field">
        <label>Strategy</label>
        <select
          className="input"
          value={strategy}
          onChange={(e) => onChange({ strategy: e.target.value })}
        >
          {STRATEGIES.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>

      <div className="section-heading">Portfolio Controls</div>
      <div className="field">
        <label>
          Risk Tolerance: <strong>{riskTolerance}</strong>
        </label>
        <input
          type="range"
          min="1"
          max="10"
          value={riskTolerance}
          onChange={(e) => onChange({ riskTolerance: Number(e.target.value) })}
        />
        <p className="field-help">
          Drag to indicate how much volatility you are comfortable accepting.
        </p>
      </div>

      <div className="field">
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={optimize}
            onChange={(e) => onChange({ optimize: e.target.checked })}
          />
          <span>Optimize portfolio weights automatically</span>
        </label>
      </div>

      <div className="field objectives-grid">
        {Object.entries(OBJECTIVE_LABELS).map(([key, label]) => (
          <label key={key} className="checkbox-pill">
            <input
              type="checkbox"
              checked={objectives[key]}
              onChange={(e) => onObjectivesChange({ [key]: e.target.checked })}
            />
            <span>{label}</span>
          </label>
        ))}
      </div>

      <div className="field grid-2">
        <div>
          <label>Initial Investment ($)</label>
          <input
            type="number"
            className="input"
            min="0"
            step="100"
            value={initialInvestment}
            onChange={(e) =>
              onChange({ initialInvestment: Number(e.target.value) })
            }
          />
        </div>
        <div>
          <label>Simulation Runs</label>
          <input
            type="number"
            className="input"
            min="100"
            max="10000"
            step="100"
            value={simulationRuns}
            onChange={(e) =>
              onChange({ simulationRuns: Number(e.target.value) })
            }
          />
        </div>
      </div>

      <div className="configuration-summary">
        <div>
          <span className="summary-label">Assets</span>
          <span className="summary-value">{symbolCount}</span>
        </div>
        <div>
          <span className="summary-label">Objectives Active</span>
          <span className="summary-value">
            {Object.values(objectives).filter(Boolean).length}
          </span>
        </div>
        <div>
          <span className="summary-label">Data Source</span>
          <span className="summary-value">{dataSource}</span>
        </div>
      </div>

      <button
        type="button"
        className="button run-button"
        onClick={onRun}
        disabled={loading}
      >
        {loading ? "Running Simulation..." : "Run Simulation"}
      </button>
    </div>
  );
}
