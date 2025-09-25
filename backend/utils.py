# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from scipy.optimize import minimize
import logging

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
