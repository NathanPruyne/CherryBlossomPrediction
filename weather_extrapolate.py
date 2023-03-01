import numpy as np

import argparse
from pathlib import Path
import os

#Use a linear fit to extrapolate temperature data, average for rainfall
def extrapolate(location, start_year, end_year, input_dir, output_dir):
    years = []
    for file in os.listdir(input_dir):
        if file.startswith(location):
            years.append(int(os.path.splitext(file)[0].split('_')[1]))
    years.sort()
    years = np.array(years)
    #Row 0: high temps (extrapolate)
    #Row 1: low temps (extrapolate)
    #Row 2: rainfall (average)
    year_data = np.zeros((len(years), 3, 180))
    for n, year in enumerate(years):
        year_data[n] = np.load(os.path.join(input_dir, location + '_' + str(year) + '.npy'))
    #year x category x day
    year_data = np.swapaxes(year_data, 0, 2)
    #day x category x year
    year_data = np.swapaxes(year_data, 0, 1)
    #category x day x year
    new_years = np.arange(start_year, end_year + 1)
    extrapolated = np.zeros((3, 180, len(new_years)))
    for channel in range(2): #Handle first 2 categories: temps
        for day, day_data in enumerate(year_data[channel]):
            fit = np.poly1d(np.polyfit(years, day_data, 1))
            extrapolated[channel][day] = fit(new_years)
    #Handle 3rd category: average
    for day, day_data in enumerate(year_data[2]):
        extrapolated[2][day] = np.average(day_data)
    extrapolated = np.swapaxes(extrapolated, 0, 2)
    # year x day x category
    extrapolated = np.swapaxes(extrapolated, 1, 2)
    # year x category x day
    np.set_printoptions(suppress = True)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    for year_num, new_data in enumerate(extrapolated):
        np.save(os.path.join(output_dir, location + '_' + str(new_years[year_num]) + '.npy'), new_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--location', type=str, help="Location to extrapolate for")
    parser.add_argument('--start_year', type=int, help='Year to start extrapolation')
    parser.add_argument('--end_year', type=int, help='Year to end extrapolation')
    parser.add_argument('--input_dir', type=Path, help="Directory with input files")
    parser.add_argument('--output_dir', type=Path, help="Directory to store outputs")

    args = parser.parse_args()

    extrapolate(args.location, args.start_year, args.end_year, args.input_dir, args.output_dir)