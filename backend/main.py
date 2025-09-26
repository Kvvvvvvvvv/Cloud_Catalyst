from fastapi import FastAPI, HTTPException
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
    calculate_prediction_metrics,
    calculate_portfolio_prediction_metrics,
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
    prediction_metrics: Optional[Dict[str, Any]] = None  # New field for metrics


class PredictionMetricsRequest(BaseModel):
    symbol: str
    prediction_days: int = Field(30, ge=1, le=90)
    test_split: float = Field(0.2, ge=0.1, le=0.5)


class PortfolioPredictionMetricsRequest(BaseModel):
    symbols: List[str]
    weights: List[float]
    prediction_days: int = Field(30, ge=1, le=90)


class PredictionMetricsResponse(BaseModel):
    symbol: str
    metrics: Dict[str, float]
    visualization_data: Dict[str, Any]
    interpretation: Dict[str, str]
    test_period_days: int
    training_period_days: int


class PortfolioPredictionMetricsResponse(BaseModel):
    portfolio_metrics: Dict[str, Any]
    individual_metrics: List[Dict[str, Any]]
    portfolio_interpretation: Dict[str, str]
    recommendation: str


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

    # Calculate prediction metrics for the portfolio
    prediction_metrics_payload = None
    try:
        prediction_metrics_payload = calculate_portfolio_prediction_metrics(
            symbols, weights, prediction_days=30
        )
    except Exception as e:
        print(f"Warning: Could not calculate prediction metrics: {e}")
        prediction_metrics_payload = None

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
        prediction_metrics=prediction_metrics_payload,
    )


# ======================= NEW PREDICTION METRICS ENDPOINTS ======================= #

@app.post("/prediction-metrics", response_model=PredictionMetricsResponse)
def api_prediction_metrics(payload: PredictionMetricsRequest):
    """
    Calculate comprehensive prediction metrics for a single stock.
    
    Returns:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error)
    - R² (Coefficient of Determination)
    - Sharpe Ratio (Financial Metric)
    - Directional Accuracy
    - Visualization data for charts
    """
    try:
        symbol = payload.symbol.upper().strip()
        metrics_result = calculate_prediction_metrics(
            symbol=symbol,
            prediction_days=payload.prediction_days,
            test_split=payload.test_split
        )
        
        return PredictionMetricsResponse(
            symbol=metrics_result['symbol'],
            metrics=metrics_result['metrics'],
            visualization_data=metrics_result['visualization_data'],
            interpretation=metrics_result['interpretation'],
            test_period_days=metrics_result['test_period_days'],
            training_period_days=metrics_result['training_period_days']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/portfolio-prediction-metrics", response_model=PortfolioPredictionMetricsResponse)
def api_portfolio_prediction_metrics(payload: PortfolioPredictionMetricsRequest):
    """
    Calculate comprehensive prediction metrics for an entire portfolio.
    
    Returns weighted portfolio metrics plus individual stock metrics.
    """
    try:
        symbols = [s.strip().upper() for s in payload.symbols if s.strip()]
        weights = payload.weights
        
        if len(symbols) != len(weights):
            raise HTTPException(status_code=400, detail="Number of symbols must match number of weights")
        
        if abs(sum(weights) - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
        
        metrics_result = calculate_portfolio_prediction_metrics(
            symbols=symbols,
            weights=weights,
            prediction_days=payload.prediction_days
        )
        
        return PortfolioPredictionMetricsResponse(
            portfolio_metrics=metrics_result['portfolio_metrics'],
            individual_metrics=metrics_result['individual_metrics'],
            portfolio_interpretation=metrics_result['portfolio_interpretation'],
            recommendation=metrics_result['recommendation']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/prediction-metrics/{symbol}")
def api_quick_prediction_metrics(symbol: str, days: int = 30):
    """
    Quick endpoint to get prediction metrics for a single symbol.
    
    Example: GET /prediction-metrics/AAPL?days=30
    """
    try:
        symbol = symbol.upper().strip()
        metrics_result = calculate_prediction_metrics(
            symbol=symbol,
            prediction_days=days,
            test_split=0.2
        )
        
        return {
            "symbol": symbol,
            "prediction_days": days,
            "metrics": metrics_result['metrics'],
            "interpretation": metrics_result['interpretation'],
            "recommendation": (
                "STRONG BUY" if metrics_result['metrics']['r_squared'] > 0.8 and 
                                metrics_result['metrics']['sharpe_ratio'] > 1 else
                "BUY" if metrics_result['metrics']['r_squared'] > 0.6 and 
                         metrics_result['metrics']['sharpe_ratio'] > 0.5 else
                "HOLD" if metrics_result['metrics']['r_squared'] > 0.4 else
                "AVOID"
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/metrics-explanation")
def api_metrics_explanation():
    """
    Get detailed explanations of all prediction metrics.
    """
    return {
        "metrics_explained": {
            "MAE": {
                "name": "Mean Absolute Error",
                "description": "Measures average magnitude of prediction errors. Lower = better.",
                "interpretation": "Shows how far off predictions are on average. A MAE of $5 means predictions are typically $5 away from actual prices.",
                "good_threshold": "< 10 for stocks, < 5 for excellent performance"
            },
            "RMSE": {
                "name": "Root Mean Squared Error",
                "description": "Penalizes larger errors more heavily than MAE. Lower = better fit.",
                "interpretation": "Emphasizes larger prediction errors. If RMSE >> MAE, the model struggles with extreme price movements.",
                "good_threshold": "< 15 for stocks, similar to MAE for consistent performance"
            },
            "MAPE": {
                "name": "Mean Absolute Percentage Error",
                "description": "Measures error as percentage of actual stock price. Useful for financial interpretability.",
                "interpretation": "Shows prediction accuracy as a percentage. 5% MAPE means predictions are typically 5% off from actual prices.",
                "good_threshold": "< 10% for trading, < 5% for excellent performance"
            },
            "R²": {
                "name": "Coefficient of Determination",
                "description": "Explains how much variance in stock prices is captured by the model. Higher (closer to 1) = better.",
                "interpretation": "Shows how well the model explains price movements. 0.8 means 80% of price variance is explained by the model.",
                "good_threshold": "> 0.6 for good models, > 0.8 for excellent models"
            },
            "Sharpe_Ratio": {
                "name": "Sharpe Ratio (Financial Metric)",
                "description": "Measures risk-adjusted return from trading strategies using the predictions. Higher = better portfolio strategy.",
                "interpretation": "Shows if prediction-based trading is profitable after accounting for risk. > 1 is good, > 2 is excellent.",
                "good_threshold": "> 1 for profitable trading, > 2 for excellent risk-adjusted returns"
            },
            "Directional_Accuracy": {
                "name": "Directional Accuracy",
                "description": "Percentage of time the model correctly predicts if price will go up or down.",
                "interpretation": "Critical for trading strategies. 60% means the model correctly predicts price direction 6 out of 10 times.",
                "good_threshold": "> 55% for useful predictions, > 65% for excellent directional accuracy"
            }
        },
        "usage_guide": {
            "for_trading": "Focus on MAPE < 10%, Sharpe Ratio > 1, and Directional Accuracy > 55%",
            "for_analysis": "Focus on R² > 0.6 and RMSE close to MAE for consistent predictions",
            "for_risk_management": "Monitor Sharpe Ratio and ensure RMSE doesn't significantly exceed MAE"
        },
        "recommended_thresholds": {
            "excellent_model": "MAE < 5, MAPE < 5%, R² > 0.8, Sharpe > 2",
            "good_model": "MAE < 10, MAPE < 10%, R² > 0.6, Sharpe > 1",
            "acceptable_model": "MAE < 15, MAPE < 15%, R² > 0.4, Sharpe > 0.5",
            "poor_model": "Any metric below acceptable thresholds"
        }
    }
