import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from interpolation import interpolate
from influence import influence
from scipy.interpolate import interp1d

def get_mass_point(z_values, weights, percentage):
    # Sort z_values and corresponding weights
    sorted_indices = np.argsort(z_values)
    sorted_z_values = np.array(z_values)[sorted_indices]
    sorted_weights = np.array(weights)[sorted_indices]

    # Compute cumulative weights
    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    # Normalize cumulative weights to create a CDF
    cdf = cumulative_weights / total_weight

    # Create an interpolation function for the CDF
    cdf_interp = interp1d(cdf, sorted_z_values, bounds_error=False, fill_value=(sorted_z_values[0], sorted_z_values[-1]))

    # Find the z value corresponding to the given percentage
    threshold_weight = (100-percentage) / 100.0
    line = cdf_interp(threshold_weight)
    
    return line

def prediction_weights(df, c, lags=30, STP_skipped=[], days_effect=False, day=None):
    phi_arr = []
    for index, c2 in enumerate(df['RWZI_AWZI_name'].unique()):
        # Only included STP's that we want
        if c2 in STP_skipped:
            continue

        X = df[df['RWZI_AWZI_name'] == c]['RNA_flow_per_100000'].tail(lags).values
        Y = df[df['RWZI_AWZI_name'] == c2]['RNA_flow_per_100000'].tail(lags).values

        if days_effect:
            latest_timepoint_c2 = df[(df['RWZI_AWZI_name'] == c2) & (df['is_interpolated'] == True)]['Date_measurement'].max()
            latest_timepoint_c2_str = latest_timepoint_c2.strftime("%Y-%m-%d")
            days_expired = (datetime.datetime.strptime(day, "%Y-%m-%d") - datetime.datetime.strptime(latest_timepoint_c2_str, "%Y-%m-%d")).days
        else:
            days_expired = 1
        phi, z = influence(X, Y, days_expired=days_expired)
        phi_arr.append((c2, phi, z))

        percentage = (index + 1) / len(df['RWZI_AWZI_name'].unique()) * 100
        print(f"{c} - {percentage:.4f}% done")

    z_values = [z for _, _, z in phi_arr]
    weights = [phi for _, phi, _ in phi_arr]

    return z_values, weights

def prediction_bounds(z_values, weights, lower_percentage=5, upper_percentage=5):
    yleft = get_mass_point(z_values, weights, percentage=100-lower_percentage)
    yright = get_mass_point(z_values, weights, percentage=upper_percentage)
    
    return yleft, yright

if __name__ == "__main__":
    file_path = "data/2024_COVID-19_rioolwaterdata.csv"
    df = pd.read_csv(file_path, delimiter=',')
    day = "2024-06-30"
    df = interpolate(df, day, splines_order=1)
    c = "Tilburg"
    lower_percentage = 5
    upper_percentage = 5

    z_values, weights = prediction_weights(df, c, lags=5, STP_skipped=[], days_effect=True, day=day)
    yleft, yright = prediction_bounds(z_values, weights, lower_percentage=lower_percentage, upper_percentage=upper_percentage)

    plt.axvline(yleft, color='r', linestyle='dashed', linewidth=1, label=f'{100-lower_percentage}% mass on the right (z={yleft:.2f})')
    plt.axvline(yright, color='r', linestyle='dashed', linewidth=1, label=f'5% mass on the right (z={upper_percentage:.2f})')
    plt.legend()

    plt.hist(z_values, weights=weights, bins=30, edgecolor='black')
    plt.xlabel('z values')
    plt.ylabel('Weight of phi values')
    plt.title('Histogram of z values weighted by phi values')
    plt.show()


