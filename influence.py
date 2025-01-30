import pandas as pd
import datetime
import numpy as np
from interpolation import interpolate
from regression import homo_regression_predictions, hetero_regression_predictions

def influence(X, Y, days_expired=1):
    homo_predictions, z_homo = homo_regression_predictions(X)
    homo_predictions = np.array(homo_predictions)
    hetero_predictions, z_hetero = hetero_regression_predictions(X, Y)
    hetero_predictions = np.array(hetero_predictions)

    X_values = np.array(X[5:])

    C = np.linalg.norm(homo_predictions - X_values) / np.linalg.norm( hetero_predictions - X_values )
    C *= 1/(days_expired**(1/2))

    return C, z_hetero

if __name__ == "__main__":
    file_path = "data/2024_COVID-19_rioolwaterdata.csv"
    df = pd.read_csv(file_path, delimiter=',')
    day = "2024-06-30"
    df = interpolate(df, day, splines_order=3)

    c1 = "Tilburg"
    c2 = "Amsterdam West"
    X = df[df['RWZI_AWZI_name'] == c1]['RNA_flow_per_100000'].tail(30).values
    Y = df[df['RWZI_AWZI_name'] == c2]['RNA_flow_per_100000'].tail(30).values

    latest_timepoint_c2 = df[(df['RWZI_AWZI_name'] == c2) & (df['is_interpolated'] == True)]['Date_measurement'].max()
    latest_timepoint_c2_str = latest_timepoint_c2.strftime("%Y-%m-%d")
    days_expired = (datetime.datetime.strptime(day, "%Y-%m-%d") - datetime.datetime.strptime(latest_timepoint_c2_str, "%Y-%m-%d")).days
    phi, z = influence(X, Y, days_expired=days_expired)


    print(phi, z)