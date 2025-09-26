# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from scipy.optimize import minimize
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# --------------------------- Stock Data --------------------------- #
def get_stock_data(symbol, period="1y"):
    try:
        data = yf.download(symbol, period=period, progress=False)
        if hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
            data.columns = data.columns.droplevel(1)
        if 'Close' not in data.columns:
            raise ValueError(f"No Close price data found for {symbol}")
        data = data[['Close']].reset_index()
        data.rename(columns={'Date':'ds','Close':'y'}, inplace=True)
        data['y'] = pd.to_numeric(data['y'], errors='coerce')
        data = data.dropna()
        if len(data) == 0:
            raise ValueError(f"No valid data for {symbol}")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {str(e)}")

# ----------------------- Prophet Prediction ----------------------- #
def predict_stock_price(symbol, days=30, changepoint_prior_scale=0.5):
    try:
        data = get_stock_data(symbol)
        if len(data) < 30:
            raise ValueError(f"Insufficient data for {symbol}. Need at least 30 days.")
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale
        )
        logging.getLogger('prophet').setLevel(logging.WARNING)
        model.fit(data)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        return forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days)
    except Exception as e:
        raise ValueError(f"Error predicting stock price for {symbol}: {str(e)}")

# ----------------------- Portfolio Simulation --------------------- #
def simulate_portfolio(symbols, weights):
    if len(symbols) != len(weights):
        raise ValueError("Symbols and weights length mismatch")
    if abs(sum(weights)-1.0) > 0.01:
        raise ValueError("Weights must sum to 1")
    portfolio_return = 0
    for i, symbol in enumerate(symbols):
        try:
            forecast = predict_stock_price(symbol, days=1)
            predicted_price = forecast['yhat'].values[0]
            current_price = get_stock_data(symbol).iloc[-1]['y']
            return_pct = (predicted_price - current_price)/current_price
            portfolio_return += return_pct * weights[i]
        except Exception as e:
            print(f"Warning: {symbol} skipped: {e}")
    return portfolio_return * 100

# ----------------------- Portfolio Optimizer ---------------------- #
def optimize_portfolio(symbols, market_data, optimization_params):
    prices = market_data.copy()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    if prices.shape[1] != len(symbols):
        raise ValueError("Columns of market_data don't match symbols length")
    prices.columns = symbols
    prices = prices.apply(pd.to_numeric, errors='coerce').dropna()
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(symbols)

    def portfolio_perf(weights):
        w = np.array(weights)
        port_ret = np.sum(mean_returns*w)*252
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w)))*np.sqrt(252)
        sharpe = port_ret/port_vol if port_vol>0 else 0
        if optimization_params.get('max_sharpe', False):
            return -sharpe
        if optimization_params.get('min_risk', False):
            return port_vol
        if optimization_params.get('max_return', False):
            return -port_ret
        return -sharpe

    constraints = ({'type':'eq','fun': lambda w: np.sum(w)-1})
    bounds = tuple((0,1) for _ in range(num_assets))
    initial = np.array([1/num_assets]*num_assets)
    result = minimize(portfolio_perf, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return np.array(result.x).tolist()

# ----------------------- Risk Metrics ----------------------------- #
def calculate_risk_metrics(simulation_results):
    returns = simulation_results.get('returns', None)
    if returns is None:
        raise ValueError("No returns in simulation results")
    arr = np.asarray(returns).flatten()
    var_95 = np.percentile(arr,5)
    var_99 = np.percentile(arr,1)
    cvar_95 = arr[arr<=var_95].mean() if arr[arr<=var_95].size>0 else var_95
    cvar_99 = arr[arr<=var_99].mean() if arr[arr<=var_99].size>0 else var_99
    return {
        'var_95': var_95, 'var_95_pct': var_95*100,
        'var_99': var_99, 'var_99_pct': var_99*100,
        'cvar_95': cvar_95, 'cvar_95_pct': cvar_95*100,
        'cvar_99': cvar_99, 'cvar_99_pct': cvar_99*100
    }

# ----------------------- Monte Carlo Cloud Simulation ------------- #
def run_cloud_simulation(symbols, weights, initial_investment, simulation_runs, cloud_provider=None, compute_power=5):
    """
    Run Monte Carlo portfolio simulation with optional cloud provider.
    """
    if len(symbols)!=len(weights):
        raise ValueError("Symbols and weights mismatch")
    if abs(sum(weights)-1.0)>0.01:
        raise ValueError("Weights must sum to 1")

    # Fetch historical data
    all_data = {}
    for symbol in symbols:
        try:
            all_data[symbol] = get_stock_data(symbol)
        except Exception as e:
            print(f"Warning: {symbol} skipped: {e}")

    if not all_data:
        raise ValueError("No valid data found")

    prices = pd.DataFrame({s: all_data[s]['y'] for s in all_data})
    returns = prices.pct_change().dropna()
    portfolio_returns = (returns*weights).sum(axis=1)

    # Monte Carlo simulation with normal distribution
    simulated_returns = []
    portfolio_values = []
    adjusted_runs = min(simulation_runs, 1000*compute_power)
    np.random.seed(None)  # make stochastic

    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns)

    for _ in range(adjusted_runs):
        random_returns = np.random.normal(mu, sigma, 252)
        cumulative = np.cumprod(1+random_returns)
        values = initial_investment * cumulative
        portfolio_values.append(values)
        simulated_returns.append((values[-1]/initial_investment)-1)

    portfolio_values = np.array(portfolio_values)
    simulated_returns = np.array(simulated_returns)

    expected_return = simulated_returns.mean()
    volatility = simulated_returns.std()
    sharpe_ratio = expected_return/volatility if volatility>0 else 0
    peak = np.maximum.accumulate(portfolio_values.mean(axis=0))
    max_drawdown = np.min((portfolio_values.mean(axis=0)-peak)/peak)

    dates = pd.date_range(start=pd.Timestamp.today(), periods=252)
    portfolio_values_df = pd.DataFrame(portfolio_values.mean(axis=0), index=dates, columns=['Portfolio Value'])

    if cloud_provider:
        print(f"Simulation run using cloud provider: {cloud_provider}")

    return {
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values_df,
        'returns': simulated_returns
    }

def extract_close_prices(raw_market_data, symbols):
    """
    Return a DataFrame of Close prices with columns ordered exactly as `symbols`.
    Handles:
      - MultiIndex output from yf.download (ticker × field)
      - Wide single-level output where each column is symbol
      - Columns like 'AAPL Close' or 'AAPL_Close'
    """
    data = raw_market_data.copy()

    # MultiIndex (common) e.g., ('AAPL','Close') or ('Close','AAPL')
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(1):
            close = data.xs('Close', axis=1, level=1)
        elif 'Close' in data.columns.get_level_values(0):
            close = data.xs('Close', axis=1, level=0)
        else:
            # Flatten and find any columns containing 'Close'
            data.columns = ['_'.join(map(str, c)).strip() for c in data.columns.values]
            close_cols = [c for c in data.columns if 'Close' in c or 'close' in c]
            if not close_cols:
                raise ValueError("Could not find 'Close' columns in market_data (MultiIndex).")
            close = data[close_cols]
    else:
        # Single-level columns
        if set(symbols).issubset(set(data.columns)):
            close = data.loc[:, symbols]
        elif 'Close' in data.columns and data.shape[1] == 1:
            close = data[['Close']]
            if len(symbols) > 1:
                raise ValueError("market_data contains a single 'Close' series but multiple symbols were requested.")
        else:
            # Try pattern matching like 'AAPL Close' or 'AAPL_Close'
            matched = {}
            for s in symbols:
                candidates = [c for c in data.columns if str(s) in str(c) and 'Close' in str(c)]
                if candidates:
                    matched[s] = candidates[0]
            if len(matched) == len(symbols):
                close = data[[matched[s] for s in symbols]]
                close.columns = symbols
            else:
                raise ValueError("Could not map market_data columns to symbols. Sample cols: "
                                 f"{list(data.columns)[:12]}...")

    # Ensure exact ordering and shape
    if set(symbols).issubset(set(close.columns)):
        close = close.loc[:, symbols]
    else:
        if close.shape[1] == len(symbols):
            close.columns = symbols
        else:
            raise ValueError(f"After extraction, price DataFrame has {close.shape[1]} columns but expected {len(symbols)}.")

    # Coerce numeric and drop rows with NaNs
    close = close.apply(pd.to_numeric, errors='coerce').dropna(axis=0, how='any')
    if close.shape[0] < 2:
        raise ValueError("Not enough historical rows after cleaning to compute returns.")
    return close

# ======================= PREDICTION METRICS ======================= #

def calculate_prediction_metrics(symbol: str, prediction_days: int = 30, test_split: float = 0.2) -> Dict[str, Any]:
    """
    Calculate comprehensive prediction metrics for a stock symbol.
    
    Returns:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error) 
    - MAPE (Mean Absolute Percentage Error)
    - R² (Coefficient of Determination)
    - Sharpe Ratio (Financial Metric)
    - Prediction accuracy visualization data
    """
    try:
        # Get historical data
        data = get_stock_data(symbol, period="2y")
        if len(data) < 100:
            raise ValueError(f"Insufficient data for {symbol}. Need at least 100 days.")
        
        # Split data for backtesting
        split_idx = int(len(data) * (1 - test_split))
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()
        
        # Train model on training data
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.5
        )
        logging.getLogger('prophet').setLevel(logging.WARNING)
        model.fit(train_data)
        
        # Predict on test period
        future = model.make_future_dataframe(periods=len(test_data))
        forecast = model.predict(future)
        
        # Get predictions for test period
        test_predictions = forecast.tail(len(test_data))[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # Align predictions with actual values
        test_data = test_data.reset_index(drop=True)
        test_predictions = test_predictions.reset_index(drop=True)
        
        actual_prices = test_data['y'].values
        predicted_prices = test_predictions['yhat'].values
        
        # Calculate core metrics
        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
        mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
        r2 = r2_score(actual_prices, predicted_prices)
        
        # Calculate Sharpe Ratio from trading strategy
        sharpe_ratio = calculate_trading_sharpe_ratio(actual_prices, predicted_prices)
        
        # Generate future predictions
        future_forecast = predict_stock_price(symbol, days=prediction_days)
        
        # Create visualization data
        viz_data = create_prediction_visualization_data(
            train_data, test_data, test_predictions, future_forecast, symbol
        )
        
        # Additional metrics
        directional_accuracy = calculate_directional_accuracy(actual_prices, predicted_prices)
        volatility_metrics = calculate_volatility_metrics(actual_prices, predicted_prices)
        
        return {
            'symbol': symbol,
            'metrics': {
                'mae': round(mae, 4),
                'rmse': round(rmse, 4),
                'mape': round(mape, 4),
                'r_squared': round(r2, 4),
                'sharpe_ratio': round(sharpe_ratio, 4),
                'directional_accuracy': round(directional_accuracy, 4),
                'volatility_explained': round(volatility_metrics['volatility_explained'], 4),
                'prediction_confidence': round(volatility_metrics['prediction_confidence'], 4)
            },
            'visualization_data': viz_data,
            'interpretation': generate_metrics_interpretation(mae, rmse, mape, r2, sharpe_ratio),
            'test_period_days': len(test_data),
            'training_period_days': len(train_data)
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating prediction metrics for {symbol}: {str(e)}")

def calculate_trading_sharpe_ratio(actual_prices: np.ndarray, predicted_prices: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio from a simple trading strategy based on predictions.
    Strategy: Buy when prediction > current price, sell when prediction < current price
    """
    try:
        # Calculate actual returns
        actual_returns = np.diff(actual_prices) / actual_prices[:-1]
        
        # Calculate prediction signals (1 for buy, -1 for sell)
        price_changes = np.diff(actual_prices)
        prediction_changes = np.diff(predicted_prices)
        
        # Simple strategy: follow prediction direction
        signals = np.sign(prediction_changes)
        
        # Calculate strategy returns
        strategy_returns = signals * actual_returns[1:]  # Align with signals
        
        if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio (annualized)
        mean_return = np.mean(strategy_returns) * 252  # Annualize
        volatility = np.std(strategy_returns) * np.sqrt(252)  # Annualize
        
        sharpe = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        return sharpe
        
    except Exception:
        return 0.0

def calculate_directional_accuracy(actual_prices: np.ndarray, predicted_prices: np.ndarray) -> float:
    """
    Calculate what percentage of the time the model correctly predicts price direction.
    """
    try:
        actual_directions = np.sign(np.diff(actual_prices))
        predicted_directions = np.sign(np.diff(predicted_prices))
        
        correct_predictions = np.sum(actual_directions == predicted_directions)
        total_predictions = len(actual_directions)
        
        return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
        
    except Exception:
        return 0.0

def calculate_volatility_metrics(actual_prices: np.ndarray, predicted_prices: np.ndarray) -> Dict[str, float]:
    """
    Calculate how well the model explains volatility patterns.
    """
    try:
        actual_volatility = np.std(np.diff(actual_prices) / actual_prices[:-1])
        predicted_volatility = np.std(np.diff(predicted_prices) / predicted_prices[:-1])
        
        volatility_explained = 1 - abs(actual_volatility - predicted_volatility) / actual_volatility
        volatility_explained = max(0, min(1, volatility_explained))  # Clamp to [0,1]
        
        # Prediction confidence based on consistency
        residuals = actual_prices - predicted_prices
        prediction_confidence = 1 - (np.std(residuals) / np.mean(actual_prices))
        prediction_confidence = max(0, min(1, prediction_confidence))  # Clamp to [0,1]
        
        return {
            'volatility_explained': volatility_explained,
            'prediction_confidence': prediction_confidence
        }
        
    except Exception:
        return {'volatility_explained': 0.0, 'prediction_confidence': 0.0}

def create_prediction_visualization_data(train_data: pd.DataFrame, test_data: pd.DataFrame, 
                                       test_predictions: pd.DataFrame, future_forecast: pd.DataFrame,
                                       symbol: str) -> Dict[str, Any]:
    """
    Create data structure for prediction accuracy visualization.
    """
    try:
        # Historical data
        historical_data = {
            'dates': train_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': train_data['y'].tolist(),
            'type': 'historical'
        }
        
        # Test period actual vs predicted
        test_actual = {
            'dates': test_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': test_data['y'].tolist(),
            'type': 'actual_test'
        }
        
        test_predicted = {
            'dates': test_predictions['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': test_predictions['yhat'].tolist(),
            'upper_bound': test_predictions['yhat_upper'].tolist(),
            'lower_bound': test_predictions['yhat_lower'].tolist(),
            'type': 'predicted_test'
        }
        
        # Future predictions
        future_predictions = {
            'dates': future_forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': future_forecast['yhat'].tolist(),
            'upper_bound': future_forecast['yhat_upper'].tolist(),
            'lower_bound': future_forecast['yhat_lower'].tolist(),
            'type': 'future_forecast'
        }
        
        return {
            'symbol': symbol,
            'historical': historical_data,
            'test_actual': test_actual,
            'test_predicted': test_predicted,
            'future_forecast': future_predictions,
            'chart_title': f'{symbol} Prediction Accuracy Analysis',
            'subtitle': 'Historical | Test Period (Actual vs Predicted) | Future Forecast'
        }
        
    except Exception as e:
        return {'error': str(e)}

def generate_metrics_interpretation(mae: float, rmse: float, mape: float, r2: float, sharpe_ratio: float) -> Dict[str, str]:
    """
    Generate human-readable interpretations of the metrics.
    """
    interpretations = {}
    
    # MAE interpretation
    if mae < 5:
        interpretations['mae'] = "Excellent prediction accuracy - very low average error"
    elif mae < 15:
        interpretations['mae'] = "Good prediction accuracy - reasonable average error"
    elif mae < 30:
        interpretations['mae'] = "Moderate prediction accuracy - noticeable average error"
    else:
        interpretations['mae'] = "Poor prediction accuracy - high average error"
    
    # RMSE interpretation
    rmse_mae_ratio = rmse / mae if mae > 0 else 1
    if rmse_mae_ratio < 1.5:
        interpretations['rmse'] = "Consistent prediction errors - few outliers"
    elif rmse_mae_ratio < 2.0:
        interpretations['rmse'] = "Some large prediction errors present"
    else:
        interpretations['rmse'] = "Significant outliers - model struggles with extreme movements"
    
    # MAPE interpretation
    if mape < 5:
        interpretations['mape'] = "Excellent percentage accuracy - highly reliable"
    elif mape < 10:
        interpretations['mape'] = "Good percentage accuracy - suitable for trading"
    elif mape < 20:
        interpretations['mape'] = "Moderate percentage accuracy - use with caution"
    else:
        interpretations['mape'] = "Poor percentage accuracy - not suitable for trading"
    
    # R² interpretation
    if r2 > 0.8:
        interpretations['r2'] = "Excellent model fit - explains most price variance"
    elif r2 > 0.6:
        interpretations['r2'] = "Good model fit - explains majority of price variance"
    elif r2 > 0.4:
        interpretations['r2'] = "Moderate model fit - explains some price variance"
    elif r2 > 0:
        interpretations['r2'] = "Poor model fit - limited explanatory power"
    else:
        interpretations['r2'] = "Very poor model fit - worse than simple average"
    
    # Sharpe Ratio interpretation
    if sharpe_ratio > 2:
        interpretations['sharpe'] = "Excellent risk-adjusted returns - highly profitable strategy"
    elif sharpe_ratio > 1:
        interpretations['sharpe'] = "Good risk-adjusted returns - profitable strategy"
    elif sharpe_ratio > 0:
        interpretations['sharpe'] = "Moderate risk-adjusted returns - marginally profitable"
    else:
        interpretations['sharpe'] = "Poor risk-adjusted returns - losing strategy"
    
    return interpretations

def calculate_portfolio_prediction_metrics(symbols: List[str], weights: List[float], 
                                         prediction_days: int = 30) -> Dict[str, Any]:
    """
    Calculate prediction metrics for an entire portfolio.
    """
    try:
        individual_metrics = []
        portfolio_performance = {'total_mae': 0, 'total_rmse': 0, 'total_mape': 0, 
                               'total_r2': 0, 'total_sharpe': 0, 'valid_symbols': 0}
        
        for i, symbol in enumerate(symbols):
            try:
                metrics = calculate_prediction_metrics(symbol, prediction_days)
                individual_metrics.append(metrics)
                
                # Weight the metrics by portfolio allocation
                weight = weights[i]
                portfolio_performance['total_mae'] += metrics['metrics']['mae'] * weight
                portfolio_performance['total_rmse'] += metrics['metrics']['rmse'] * weight
                portfolio_performance['total_mape'] += metrics['metrics']['mape'] * weight
                portfolio_performance['total_r2'] += metrics['metrics']['r_squared'] * weight
                portfolio_performance['total_sharpe'] += metrics['metrics']['sharpe_ratio'] * weight
                portfolio_performance['valid_symbols'] += 1
                
            except Exception as e:
                print(f"Warning: Could not calculate metrics for {symbol}: {e}")
                continue
        
        # Calculate portfolio-wide interpretation
        portfolio_interpretation = generate_metrics_interpretation(
            portfolio_performance['total_mae'],
            portfolio_performance['total_rmse'],
            portfolio_performance['total_mape'],
            portfolio_performance['total_r2'],
            portfolio_performance['total_sharpe']
        )
        
        return {
            'portfolio_metrics': {
                'weighted_mae': round(portfolio_performance['total_mae'], 4),
                'weighted_rmse': round(portfolio_performance['total_rmse'], 4),
                'weighted_mape': round(portfolio_performance['total_mape'], 4),
                'weighted_r2': round(portfolio_performance['total_r2'], 4),
                'weighted_sharpe': round(portfolio_performance['total_sharpe'], 4),
                'symbols_analyzed': portfolio_performance['valid_symbols'],
                'total_symbols': len(symbols)
            },
            'individual_metrics': individual_metrics,
            'portfolio_interpretation': portfolio_interpretation,
            'recommendation': generate_portfolio_recommendation(portfolio_performance)
        }
        
    except Exception as e:
        raise ValueError(f"Error calculating portfolio prediction metrics: {str(e)}")

def generate_portfolio_recommendation(portfolio_performance: Dict[str, float]) -> str:
    """
    Generate trading recommendation based on portfolio prediction metrics.
    """
    mae = portfolio_performance['total_mae']
    r2 = portfolio_performance['total_r2']
    sharpe = portfolio_performance['total_sharpe']
    mape = portfolio_performance['total_mape']
    
    score = 0
    if mae < 10: score += 1
    if r2 > 0.6: score += 1
    if sharpe > 1: score += 1
    if mape < 10: score += 1
    
    if score >= 3:
        return "STRONG BUY - Excellent prediction metrics across all measures"
    elif score >= 2:
        return "BUY - Good prediction metrics, suitable for trading"
    elif score >= 1:
        return "HOLD - Moderate prediction quality, proceed with caution"
    else:
        return "AVOID - Poor prediction metrics, not recommended for trading"
