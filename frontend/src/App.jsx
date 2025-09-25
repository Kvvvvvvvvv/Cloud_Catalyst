import React, { useState } from "react";
import ConfigPanel from "./components/ConfigPanel";
import PortfolioSummary from "./components/PortfolioSummary";
import Charts from "./components/Charts";
import AllocationTable from "./components/AllocationTable";
import MarketOverview from "./components/MarketOverview";
import { runSimulation } from "./api";

const defaultObjectives = {
  max_return: true,
  min_risk: true,
  max_sharpe: true,
  esg_focus: false,
};

const initialState = {
  symbolsInput: "AAPL,MSFT,GOOG,AMZN,TSLA",
  weightsInput: "",
  optimize: true,
  initialInvestment: 10000,
  simulationRuns: 1000,
  dataSource: "Yahoo Finance",
  strategy: "Balanced Growth",
  timeHorizon: "1 Year",
  riskTolerance: 5,
  objectives: { ...defaultObjectives },
};

export default function App() {
  const [state, setState] = useState(initialState);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const onChange = (patch) => setState((prev) => ({ ...prev, ...patch }));
  const onObjectivesChange = (patch) =>
    setState((prev) => ({
      ...prev,
      objectives: { ...prev.objectives, ...patch },
    }));

  const onRun = async () => {
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const symbols = state.symbolsInput
        .split(",")
        .map((s) => s.trim().toUpperCase())
        .filter(Boolean);

      if (!symbols.length) {
        throw new Error("Please provide at least one ticker symbol.");
      }

      const payload = {
        symbols,
        weights_input: state.weightsInput,
        optimize: state.optimize,
        initial_investment: Number(state.initialInvestment),
        simulation_runs: Number(state.simulationRuns),
        data_source: state.dataSource,
        strategy: state.strategy,
        time_horizon: state.timeHorizon,
        risk_tolerance: Number(state.riskTolerance),
        optimization_options: {
          max_return: state.objectives.max_return,
          min_risk: state.objectives.min_risk,
          max_sharpe: state.objectives.max_sharpe,
          esg_focus: state.objectives.esg_focus,
          risk_tolerance: Number(state.riskTolerance),
        },
      };

      const resp = await runSimulation(payload);
      setResult(resp.data);
    } catch (err) {
      console.error(err);
      setError(
        err?.response?.data?.detail ||
          err.message ||
          "Unable to run simulation."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="hero">
        <span className="hero-badge">Preview</span>
        <h1 className="hero-title">CloudCatalyst</h1>
        <p className="hero-subtitle">
          Multi-Objective Financial Strategy Simulation &amp; Optimization
        </p>
      </header>

      <div className="app-body">
        <aside className="side-panel">
          <ConfigPanel
            state={state}
            onChange={onChange}
            onObjectivesChange={onObjectivesChange}
            onRun={onRun}
            loading={loading}
          />
          <MarketOverview metrics={result?.market_overview} />
        </aside>

        <main className="content-area">
          {loading && (
            <div className="card loading-card">
              Running simulation — this may take a minute...
            </div>
          )}

          {error && <div className="card error-card">{error}</div>}

          {result ? (
            <>
              <PortfolioSummary summary={result} />
              <Charts
                portfolioValues={result.portfolio_values}
                predictions={result.predictions}
              />
              <AllocationTable allocation={result.allocation} />
            </>
          ) : (
            !loading && (
              <div className="card empty-card">
                Configure your scenario on the left and run a simulation to
                explore allocation insights, predictive analytics, and risk
                metrics.
              </div>
            )
          )}
        </main>
      </div>
    </div>
  );
}
