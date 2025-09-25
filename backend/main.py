from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import pandas as pd
import yfinance as yf
from utils import (
    predict_stock_price,
    optimize_portfolio,
    run_cloud_simulation,
    calculate_risk_metrics,
    get_stock_data,
)


app = FastAPI(title="CloudCatalyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- Pydantic models ----------------------
class OptimizeOptions(BaseModel):
    max_return: bool = True
    min_risk: bool = True
    max_sharpe: bool = True
    esg_focus: bool = False
    risk_tolerance: int = 5


class RunSimulationRequest(BaseModel):
    symbols: List[str]
    weights_input: Optional[str] = Field(None, description="Comma-separated weights or empty to auto-optimize")
    optimize: bool = True
    initial_investment: float = 10000.0
    simulation_runs: int = 1000
    data_source: str = "Yahoo Finance"
    strategy: str = "Balanced Growth"
    time_horizon: str = "1 Year"
    risk_tolerance: int = Field(5, ge=1, le=10)
    optimization_options: OptimizeOptions = Field(default_factory=OptimizeOptions)


class RunSimulationResponse(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    portfolio_values: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    allocation: List[Dict[str, Any]]
    risk_metrics: Optional[Dict[str, Any]] = None
    symbols: List[str]
    weights: List[float]
    strategy: Optional[str] = None
    time_horizon: Optional[str] = None
    risk_tolerance: Optional[int] = None
    data_source: Optional[str] = None
    market_overview: Optional[List[Dict[str, Any]]] = None
    optimization_objectives: Optional[Dict[str, Any]] = None


def _model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _parse_weights(weights_input: Optional[str], symbols: List[str]) -> Optional[List[float]]:
    if not weights_input:
        return None
    try:
        weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="Weights must be numeric values.")
    if len(weights) != len(symbols):
        raise HTTPException(status_code=400, detail="Number of weights must match number of symbols.")
    total = sum(weights)
    if total <= 0:
        raise HTTPException(status_code=400, detail="Weights must sum to a positive value.")
    normalized = [w / total for w in weights]
    return normalized


def _build_market_prices(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        data = get_stock_data(symbol, period="2y")
        if data.empty:
            raise ValueError(f"No price data available for {symbol}.")
        frame = data[["ds", "y"]].rename(columns={"ds": "date", "y": symbol}).set_index("date")
        frames.append(frame)
    merged = pd.concat(frames, axis=1).dropna()
    if merged.empty:
        raise ValueError("Insufficient overlapping historical data for the selected symbols.")
    return merged


def _build_predictions(symbols: List[str], days: int = 30) -> List[Dict[str, Any]]:
    series = []
    for symbol in symbols:
        try:
            forecast = predict_stock_price(symbol, days=days)
            dates = pd.to_datetime(forecast["ds"]).dt.strftime("%Y-%m-%d").tolist()
            series.append(
                {
                    "symbol": symbol,
                    "ds": dates,
                    "yhat": forecast["yhat"].astype(float).tolist(),
                    "yhat_lower": forecast["yhat_lower"].astype(float).tolist(),
                    "yhat_upper": forecast["yhat_upper"].astype(float).tolist(),
                }
            )
        except Exception as exc:  # pragma: no cover - best-effort forecast
            series.append({"symbol": symbol, "ds": [], "yhat": [], "error": str(exc)})
    return series


def _index_snapshot(ticker: str, label: str) -> Dict[str, Any]:
    try:
        history = yf.download(ticker, period="5d", interval="1d", progress=False)
        if history.empty or "Close" not in history:
            return {"label": label, "ticker": ticker, "value": None, "change_pct": None}
        close = history["Close"].astype(float)
        latest = float(close.iloc[-1])
        previous = float(close.iloc[-2]) if close.size > 1 else latest
        change_pct = ((latest - previous) / previous * 100.0) if previous else 0.0
        return {
            "label": label,
            "ticker": ticker,
            "value": round(latest, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as exc:  # pragma: no cover - best effort
        return {"label": label, "ticker": ticker, "error": str(exc)}


def _generate_market_overview() -> List[Dict[str, Any]]:
    indices = {
        "^GSPC": "S&P 500",
        "^IXIC": "NASDAQ",
        "^DJI": "Dow Jones",
        "^VIX": "VIX",
    }
    return [_index_snapshot(ticker, label) for ticker, label in indices.items()]


# ---------------------- Helper endpoints ----------------------
# ---------------------- Helper endpoints ----------------------
@app.get("/predict/{symbol}")
def api_predict(symbol: str, days: int = 30):
    """
    Predict future stock prices for `symbol`.

    Input:
        - `symbol` (path parameter, str): stock ticker.
        - `days` (query parameter, int): number of days to forecast.

    Output:
        JSON list with keys ['ds','yhat','yhat_lower','yhat_upper'].

    Example use case:
        GET /predict/AAPL?days=30
    """
    try:
        df = predict_stock_price(symbol, days=days)
        res = df.reset_index(drop=True).to_dict(orient='records')
        return {"symbol": symbol, "forecast": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/run-simulation", response_model=RunSimulationResponse)
def api_run_simulation(payload: RunSimulationRequest):
    symbols = [s.strip().upper() for s in payload.symbols if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="Please provide at least one ticker symbol.")

    weights = _parse_weights(payload.weights_input, symbols)
    objectives_model = payload.optimization_options
    if hasattr(objectives_model, "risk_tolerance"):
        objectives_model.risk_tolerance = payload.risk_tolerance

    try:
        market_prices = _build_market_prices(symbols)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if payload.optimize or weights is None:
        try:
            optimization_params = _model_to_dict(objectives_model)
            optimized = optimize_portfolio(symbols, market_prices, optimization_params)
            weights = [float(w) for w in optimized]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Optimization failed: {exc}")

    # Ensure weights are valid
    weights = _parse_weights(",".join(str(w) for w in weights), symbols)
    if weights is None:
        weights = [1 / len(symbols)] * len(symbols)

    try:
        simulation = run_cloud_simulation(
            symbols=symbols,
            weights=weights,
            initial_investment=payload.initial_investment,
            simulation_runs=payload.simulation_runs,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Simulation failed: {exc}")

    portfolio_df = simulation.get("portfolio_values")
    if portfolio_df is None or portfolio_df.empty:
        raise HTTPException(status_code=500, detail="Simulation returned no portfolio values.")

    portfolio_payload = {
        "dates": [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in portfolio_df.index],
        "values": portfolio_df.iloc[:, 0].astype(float).tolist(),
    }

    allocation_payload = [
        {"asset": symbol, "allocation_pct": float(weight) * 100.0}
        for symbol, weight in zip(symbols, weights)
    ]

    predictions_payload = _build_predictions(symbols)
    risk_payload = None
    try:
        risk_payload = calculate_risk_metrics(simulation)
    except Exception:
        risk_payload = None

    market_overview = _generate_market_overview()
    optimization_objectives = _model_to_dict(objectives_model)

    return RunSimulationResponse(
        expected_return=float(simulation.get("expected_return", 0.0)),
        volatility=float(simulation.get("volatility", 0.0)),
        sharpe_ratio=float(simulation.get("sharpe_ratio", 0.0)),
        max_drawdown=float(simulation.get("max_drawdown", 0.0)),
        portfolio_values=portfolio_payload,
        predictions=predictions_payload,
        allocation=allocation_payload,
        risk_metrics=risk_payload,
        symbols=symbols,
        weights=weights,
        strategy=payload.strategy,
        time_horizon=payload.time_horizon,
        risk_tolerance=payload.risk_tolerance,
        data_source=payload.data_source,
        market_overview=market_overview,
        optimization_objectives=optimization_objectives,
    )
