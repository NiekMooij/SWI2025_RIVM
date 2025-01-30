import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import sys

from interpolation import interpolate
from implementation import new_measurements, check_for_outliers

if __name__ == "__main__":
    file_path = "data/2024_COVID-19_rioolwaterdata.csv"
    df = pd.read_csv(file_path, delimiter=',')
    day = "2024-06-30"
    measurements = new_measurements(df, day)
    measurements = measurements[:14]
    lags = 2 # Time windows over which we train the regression coefficients

    df = interpolate(df, day, splines_order=1)

    measurements = [ m for m in measurements if m['RWZI_AWZI_name'] in ["Apeldoorn", "Tilburg", "Susteren", "Steenwijk"] ]
    data = check_for_outliers(measurements, df, lags=lags, days_effect=True, day=day)

    num_plots = len(data)
    # grid_size = int(num_plots**0.5) + (1 if num_plots % int(num_plots**0.5) != 0 else 0)
    # grid_size = 4
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(13.33, 7.5))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    for i, m in enumerate(data):
        ax = axes[i]

        ax.text(-0.1, 1.2, chr(97 + i), transform=ax.transAxes, fontsize=18, va='top')

        z_values, weights, yleft, yright = m['z_values'], m['weights'], m['yleft'], m['yright']
        c = m['RWZI_AWZI_name']

        if c not in ["Apeldoorn", "Tilburg", "Riel", "Nieuwveer", "Susteren", "Steenwijk"]:
            continue

        ax.axvline(yleft, color='black', linestyle='dashed', linewidth=2)
        ax.axvline(yright, color='black', linestyle='dashed', linewidth=2)

        ax.axvline(m['RNA_flow_per_100000'], color='r', linestyle='dashed', linewidth=2, label=f'Measured')
        ax.legend()

        ax.hist(z_values, weights=weights, bins=30, edgecolor='black', density=True)
        ax.set_xlabel('Projection', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(f'{m['RWZI_AWZI_name']}', fontsize=14)
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)

   
        
    # save_name = os.path.join('figures/analysis.pdf')
    # plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')
    save_name = os.path.join('figures/analysis.png')
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='png')
    save_name = os.path.join('figures/analysis.pdf')
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.1, format='pdf')
    plt.show()