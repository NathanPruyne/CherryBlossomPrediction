import numpy as np
import pandas as pd

import argparse
import os

def preprocess(weather_file):
    #Load from a CSV and get useful columns
    weather_df = pd.read_csv(weather_file)
    weather_df['datetime'] = pd.to_datetime(weather_df['time'])
    weather_df['dayofyear'] = weather_df['datetime'].dt.dayofyear
    weather_df['year'] = weather_df['datetime'].dt.year
    weather_df = weather_df[weather_df.dayofyear <= 180] #Only January to June
    for year in range(weather_df['year'].min(), weather_df['year'].max()):
        year_df = weather_df[weather_df['year'] == year].copy()
        year_df.sort_values('dayofyear', inplace=True)
        rain_cumulative = year_df['rain_sum (mm)'].cumsum()
        year_df['cum_rain'] = rain_cumulative #Use cumulative rainfall
        year_df = year_df[['temperature_2m_max (°C)', 'temperature_2m_min (°C)', 'cum_rain']] #Get relevant columns
        name = os.path.splitext(os.path.basename(weather_file))[0] + '_' + str(year) + '.npy'
        if year_df.isnull().values.any(): #Make sure no NaNs
            print(f"{name} has bad data")
            continue
        year_np = year_df.to_numpy().T
        if not os.path.isdir('data/weather_cache/'):
            os.makedirs('data/weather_cache/')
        np.save('data/weather_cache/' + name, year_np)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('weather_file', help="Weather CSV to process")

    args = parser.parse_args()

    preprocess(args.weather_file)