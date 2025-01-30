import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import sys

def interpolate(df, day, splines_order=3):
    """
    Interpolates missing data in the DataFrame up to a specified day.

    Parameters:
    df (pd.DataFrame): DataFrame with 'Date_measurement' and 'RWZI_AWZI_name' columns.
    day (str): End date for interpolation in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame with interpolated values and 'is_interpolated' column.
    """
    df = df[df['Date_measurement'] < day]
    df['Date_measurement'] = pd.to_datetime(df['Date_measurement'])

    interpolated_dfs = []
    for name, group in df.groupby('RWZI_AWZI_name'):
        group = group.set_index('Date_measurement').reindex(pd.date_range(start=group['Date_measurement'].min(), end=pd.to_datetime(day) - pd.Timedelta(days=1)))
        group['is_interpolated'] = group['RNA_flow_per_100000'].isna()
        group = group.reset_index().rename(columns={'index': 'Date_measurement'})
        group = group.reset_index().rename(columns={'level_0': 'Date_measurement_temp'})
        group = group.drop(columns=['Date_measurement'], errors='ignore')
        group = group.reset_index().rename(columns={'Date_measurement_temp': 'Date_measurement'})
        group = group.drop(columns=['level_0'], errors='ignore')
        group = group.drop(columns=['index'], errors='ignore')

        group['RWZI_AWZI_name'] = name
        
        group = group.set_index('Date_measurement')
        if not isinstance(group.index, pd.DatetimeIndex):
            group.index = pd.to_datetime(group.index)

        group = group.resample('D').interpolate(method='spline', order=splines_order)
        group = group.reset_index().rename(columns={'index': 'Date_measurement'})
    
        interpolated_dfs.append(group)

    df = df.sort_values(by='Date_measurement')
    df = pd.concat(interpolated_dfs)

    return df

if __name__ == "__main__":
    file_path = "data/2024_COVID-19_rioolwaterdata.csv"
    df = pd.read_csv(file_path, delimiter=',')
    day = "2024-06-30"
    c = "Utrecht"
    df = interpolate(df, day, splines_order=2)
    df_plot = df[df['RWZI_AWZI_name'] == c]
    df_plot = df_plot[(df_plot['Date_measurement'] > '2024-05-01') & (df_plot['Date_measurement'] < '2024-06-10')]

    plt.figure(figsize=(10, 6))
    interpolated = df_plot['is_interpolated']
    plt.plot(df_plot['Date_measurement'][~interpolated], df_plot['RNA_flow_per_100000'][~interpolated], 'ro', label='Original Data', markersize=4)
    plt.plot(df_plot['Date_measurement'][interpolated], df_plot['RNA_flow_per_100000'][interpolated], 'bo', label='Interpolated Data', markersize=3)
    # plt.plot(df_plot['Date_measurement'], df_plot['RNA_flow_per_100000'], 'b-', alpha=0.5)
    plt.legend()
    plt.title(f'Interpolated Data for {c}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)

    save_name = os.path.join(sys.path[0], 'figures/interpolated_data.png')
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png', transparent=True)
    save_name = os.path.join(sys.path[0], 'figures/interpolated_data.pdf')
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf', transparent=True)

    plt.show()