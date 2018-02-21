""" Portfolio Optimizer """
import os
import pandas as pd
import matplotlib.pyplot as plt  # optional
import numpy as np
import scipy.optimize as spo

### STEPS ###
#1. Get stocks data
#2. Normalize data
#3. Multiply by allocations
#_. Multiply by starting value (optional)
#4. Sum daily portfolio value
#5. Compute daily returns of the portfolio (exclude starting zero)
#6. Compute Sharpe ratio of portfolio daily returns
#7. Optimize portfolio allocations by maximizing Sharpe ratio

def get_data(symbols, dates=None, base_dir='stocks_data'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    
    def symbol_to_path(symbol, base_dir=base_dir):
        """Return CSV file path given ticker symbol."""
        return os.path.join(base_dir, f'{str(symbol)}.csv')

    # Get data from all files
    if dates:   df = pd.DataFrame(index=dates)
    else:       df = pd.DataFrame()
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp, how='outer')

    # Fill missing values in data frame, in place. 
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    # Return data as a DataFrame
    return df


def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


def compute_daily_returns(df):
    """Compute and return the daily return values."""
    return df[1:] / df[:-1].values - 1

def compute_portfolio_daily_returns(stocks, allocations):
    """Compute and return the daily portfolio returns."""
    #3. Multiply by allocations
    portfolio = stocks*allocations
    #4. Sum daily portfolio value
    portfolio_daily_sum = portfolio.sum(axis=1)
    # Return daily returns of the portfolio (exclude starting zero)
    # Formula: df[1:] / df[:-1].values - 1
    return compute_daily_returns(portfolio_daily_sum)


def normalize(df):
    """Compute and return normalized dataframe."""
    return df / df.values[0,:]


def optimize_portfolio(stocks):
    """Compute and return optimized portfolio allocations."""
    allocs = np.array([1/len(list(stocks)) for stock in list(stocks)])

    def portfolio_sharpe_ratio(allocs, stocks=stocks):
        daily_safe_returns = 0
        #5. Compute daily returns of the portfolio (exclude starting zero)
        daily_returns = compute_portfolio_daily_returns(stocks, allocs)
        #6. Compute Sharpe ratio of portfolio daily returns
        return (daily_returns - daily_safe_returns).mean() / daily_returns.std()

    def portfolio_sharpe_ratio_inverse(allocs):
        """ Returns inverse of Sharpe ratio of the portfolio """
        return 1 / portfolio_sharpe_ratio(allocs)

    #7. Optimize portfolio allocations by maximizing Sharpe ratio (minimizing it's inverse)
    bnds = [(0, 1) for stock in list(stocks)]
    cons = ({'type': 'eq', 'fun': lambda x:  1. - sum(x)})
    optimization = spo.minimize(portfolio_sharpe_ratio_inverse, allocs, 
        method='SLSQP', bounds=bnds, constraints=cons, options={'disp': False})
    # Return optimized portfolio allocations
    return optimization.x


def run_optimization():
    #1. Get stocks data
    symbols = ['SPY','AAPL', 'GOOG', 'AMZN', 'TWTR', 'FB', 'TSLA', 'BTC-USD']
    df = get_data(symbols)

    #2. Normalize data
    normalized_data = normalize(df)
    
    #3-7. Optimize portfolio allocations
    optimized_allocations = optimize_portfolio(normalized_data)
    
    #Show pretty result
    optimized_allocations = pd.DataFrame(optimized_allocations, index=symbols, columns=['Allocations'])
    optimized_allocations['Allocations'] = optimized_allocations['Allocations'].map('{:,.2f}'.format)
    print(optimized_allocations)


if __name__ == "__main__":
    run_optimization()