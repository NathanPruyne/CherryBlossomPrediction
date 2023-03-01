import numpy as np
import pandas as pd
from scipy import signal

import argparse
import os

def preprocess(blossom_file):
    blossom_df = pd.read_csv(blossom_file)
    gaussian = signal.gaussian(11, std=2) #We soften with a Gaussian to reduce the loss hit of using a one-hot
    for year, bloom_doy in zip(blossom_df['year'], blossom_df['bloom_doy']):
        center = bloom_doy - 1 #Align to 0 center array
        full_arr = np.zeros(180)
        if center < 5: #Account for if not whole Gaussian will fit
            full_arr[:11 - center] = gaussian[center:]
        elif center > 174:
            full_arr[center:] = gaussian[:11 - center]
        else:
            full_arr[center - 5:center + 6] = gaussian
        name = os.path.splitext(os.path.basename(blossom_file))[0] + '_' + str(year) + '.npy'
        if not os.path.isdir('data/blossom_cache'):
            os.makedirs('data/blossom_cache')
        np.save('data/blossom_cache/' + name, full_arr)
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('blossom_file', help="Blossom CSV to process")

    args = parser.parse_args()

    preprocess(args.blossom_file)