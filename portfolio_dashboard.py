import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import BytesIO
import base64

# Use a non-interactive backend to avoid threading issues with Flask
import matplotlib
matplotlib.use('Agg')

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to fetch data and perform optimization
def get_optimized_portfolio(start_date, end_date, method):
    tickers = [
        'SAP.DE', 'SIE.DE', 'ASML.AS', 'VOW3.DE', 'AIR.PA', 'BNP.PA',
        'HSBA.L', 'OR.PA', 'NESN.SW', 'NOVN.SW', 'UNA.AS', 'BAS.DE',
        'SAN.PA', 'BARC.L', 'AZN.L', 'AD.AS', 'BMW.DE', 'ENEL.MI'
    ]
    
    # Fetch data
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()

    # Define initial weights and constraints
    initial_weights = np.array([1/len(tickers)] * len(tickers))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len(tickers)))

    if method == 'Optimized Portfolio':
        # Sharpe Ratio optimization
        def sharpe_ratio(weights, returns):
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
            return -portfolio_return / portfolio_volatility
        
        optimized = minimize(sharpe_ratio, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
        
    elif method == 'Optimized MRisk Portfolio':
        # Minimize risk optimization
        def portfolio_volatility(weights, returns):
            return np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        optimized = minimize(portfolio_volatility, initial_weights, args=(returns,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimized_weights = optimized.x
    portfolio_returns = returns.dot(optimized_weights)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    return cumulative_returns, optimized_weights, tickers

# Function to generate the plot
def generate_plot(cumulative_returns):
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Portfolio Cumulative Returns')
    plt.title('Portfolio Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.tight_layout()

    # Save plot to a PNG image in memory
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    return encoded_image

# App layout
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Portfolio Optimization Dashboard"), className="mb-4")),
    dbc.Row([
        dbc.Col([
            html.Label("Select Start Date"),
            dcc.DatePickerSingle(
                id='start-date-picker',
                min_date_allowed=pd.to_datetime('2000-01-01').strftime('%Y-%m-%d'),
                max_date_allowed=pd.to_datetime('today').strftime('%Y-%m-%d'),
                initial_visible_month=pd.to_datetime('2020-01-01').strftime('%Y-%m-%d'),
                date=pd.to_datetime('2020-01-01').strftime('%Y-%m-%d')
            ),
        ], width=4),
        dbc.Col([
            html.Label("Select End Date"),
            dcc.DatePickerSingle(
                id='end-date-picker',
                min_date_allowed=pd.to_datetime('2000-01-01').strftime('%Y-%m-%d'),
                max_date_allowed=pd.to_datetime('today').strftime('%Y-%m-%d'),
                initial_visible_month=pd.to_datetime('today').strftime('%Y-%m-%d'),
                date=pd.to_datetime('today').strftime('%Y-%m-%d')
            ),
        ], width=4),
        dbc.Col([
            html.Label("Select Optimization Method"),
            dcc.Dropdown(
                id='method-dropdown',
                options=[
                    {'label': 'Optimized Portfolio (Sharpe Ratio)', 'value': 'Optimized Portfolio'},
                    {'label': 'Optimized MRisk Portfolio (Min Risk)', 'value': 'Optimized MRisk Portfolio'}
                ],
                value='Optimized Portfolio'
            ),
        ], width=4)
    ]),
    dbc.Row(dbc.Col(html.Div(id='plot-container'), className="mt-4")),
    dbc.Row(dbc.Col(html.Div(id='weights-container'), className="mt-4"))
])

# Callbacks
@app.callback(
    [Output('plot-container', 'children'),
     Output('weights-container', 'children')],
    [Input('start-date-picker', 'date'),
     Input('end-date-picker', 'date'),
     Input('method-dropdown', 'value')]
)
def update_output(start_date, end_date, method):
    if start_date and end_date and method:
        cumulative_returns, optimized_weights, tickers = get_optimized_portfolio(start_date, end_date, method)
        encoded_image = generate_plot(cumulative_returns)
        
        weights_df = pd.DataFrame({"Ticker": tickers, "Weight": optimized_weights})
        weights_table = dbc.Table.from_dataframe(weights_df, striped=True, bordered=True, hover=True)

        return html.Img(src='data:image/png;base64,{}'.format(encoded_image)), weights_table
    return None, None

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
