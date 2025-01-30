import pandas as pd
import datetime

from prediction import prediction_weights, prediction_bounds
from interpolation import interpolate

def new_measurements(df, day):
    new_measurements = df[df['Date_measurement'] == day]
    new_measurements = new_measurements[['RWZI_AWZI_name', 'RNA_flow_per_100000']]
    new_measurements = list(new_measurements.itertuples(index=False, name=None))
    new_measurements = [{'RWZI_AWZI_name': item[0], 'RNA_flow_per_100000': item[1]} for item in new_measurements]

    return new_measurements 

def check_for_outliers(measurements, df, lower_percentage=5, upper_percentage=5, lags=30, days_effect=False, day=None):
    data = []
    for m in measurements:
        c = m['RWZI_AWZI_name']
        y = m['RNA_flow_per_100000']
    
        z_values, weights = prediction_weights(df, c, lags=lags, STP_skipped=[], days_effect=days_effect, day=day)
        yleft, yright = prediction_bounds(z_values, weights, lower_percentage=lower_percentage, upper_percentage=upper_percentage)

        m['z_values'] = z_values
        m['weights'] = weights
        m['yleft'] = yleft
        m['yright'] = yright
        
        if y < yleft or y > yright:
            m['outlier'] = True
        else:
            m['outlier'] = False

        data.append(m)

    return data

if __name__ == "__main__":
    file_path = "data/2024_COVID-19_rioolwaterdata.csv"
    df = pd.read_csv(file_path, delimiter=',')
    day = "2024-06-30"
    measurements = new_measurements(df, day)
    lags = 14

    # Only take first 5 elements to see what's happening
    measurements = measurements[:5]

    df = interpolate(df, day, splines_order=1)
    data = check_for_outliers(measurements, df, lags=lags, days_effect=True, day=day)

    print(data)