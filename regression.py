import pandas as pd
import datetime
from interpolation import interpolate
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import sys

def homo_regression_predictions(X, n_lags=5):
    # Check if the input data is valid
    if len(X) <= n_lags:
        raise ValueError("Length of X should be greater than n_lags.")
    
    # Fit an AR model (choosing lag=n_lags)
    model = AutoReg(X, lags=n_lags)
    model_fit = model.fit()
    
    # Predict the next values of X (given the last n_lags points of X)
    predictions = model_fit.predict(start=n_lags, end=len(X)-1)
    z = model_fit.predict(start=len(X), end=len(X))
    
    return predictions, z[0]

def hetero_regression_predictions(X, Y, n_lags=5):
    # Check if the input data is valid
    if len(X) <= n_lags or len(Y) <= n_lags:
        raise ValueError("Length of X and Y should be greater than n_lags.")
    
    # Initialize the regression model
    model = LinearRegression()
    
    # Create lagged features for both X and Y
    X_lagged = []
    Y_lagged = []
    target = []
    for i in range(n_lags, len(X)):
        X_lagged.append(X[i-n_lags:i])  # previous n_lags points of X
        Y_lagged.append(Y[i-n_lags:i])  # previous n_lags points of Y
        target.append(X[i])  # target value of X at time i

    # Convert to numpy arrays for fitting the model
    X_lagged = np.array(X_lagged)
    Y_lagged = np.array(Y_lagged)
    target = np.array(target)
    features = np.hstack((X_lagged, Y_lagged))

    # Fit the regression model
    model.fit(features, target)
    predictions = model.predict(features)
    
    # Predict the next value of X (given the last n_lags points of X and Y)
    z = model.predict(np.hstack((X[-n_lags:], Y[-n_lags:])).reshape(1, -1))
    
    return predictions, z[0]

if __name__ == "__main__":
    file_path = "data/2024_COVID-19_rioolwaterdata.csv"
    df = pd.read_csv(file_path, delimiter=',')
    day = "2024-06-30"
    df = interpolate(df, day, splines_order=2)
    c = "Tilburg"
    c2 = "Utrecht"
    df = df[(df['Date_measurement'] > '2024-05-01') & (df['Date_measurement'] < '2024-06-10')]

    X = df[df['RWZI_AWZI_name'] == c]['RNA_flow_per_100000'].tail(30).values
    Y = df[df['RWZI_AWZI_name'] == c2]['RNA_flow_per_100000'].tail(30).values

    homo_predictions, z = homo_regression_predictions(X)
    hetero_predictions, z = hetero_regression_predictions(X, Y)
    
    plt.figure(figsize=(10, 6))

    df_plot = df[df['RWZI_AWZI_name'] == c]
    # df_plot = df_plot[df_plot['Date_measurement'] < '2024-06-23']

    interpolated = df_plot['is_interpolated']
    # plt.plot(df_plot['Date_measurement'][~interpolated][:-8], df_plot['RNA_flow_per_100000'][~interpolated][:-8], 'ro', markersize=4)
    # plt.plot(df_plot['Date_measurement'][interpolated][:-8], df_plot['RNA_flow_per_100000'][interpolated][:-8], 'bo', markersize=2)
    # plt.plot(df_plot['Date_measurement'][-25:-8], homo_predictions[:-8], label='Predictions A', linestyle='--')
    # plt.plot(df_plot['Date_measurement'][-25:-8], hetero_predictions[:-8], label='Predictions A & B', linestyle='--')

    plt.plot(df_plot['Date_measurement'][~interpolated], df_plot['RNA_flow_per_100000'][~interpolated], 'ro', markersize=4)
    plt.plot(df_plot['Date_measurement'][interpolated], df_plot['RNA_flow_per_100000'][interpolated], 'bo', markersize=2)
    plt.plot(df_plot['Date_measurement'][-25:], homo_predictions, label='Predictions A', linestyle='--')
    plt.plot(df_plot['Date_measurement'][-25:], hetero_predictions, label='Predictions A & B', linestyle='--')

    plt.legend()
    plt.xlabel('Date')
    plt.title(f'{c}')
    plt.ylabel('Value')
    plt.grid(True)
    # plt.xlim(pd.Timestamp('2024-06-01'), df_plot['Date_measurement'].max() + pd.Timedelta(days=2))

    save_name = os.path.join(sys.path[0], 'figures/regression.png')
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png', transparent=True)
    save_name = os.path.join(sys.path[0], 'figures/regression.pdf')
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf', transparent=True)

    plt.show()