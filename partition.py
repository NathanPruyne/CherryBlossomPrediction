import json
import os
import random

if __name__ == '__main__':
    blossom_files = os.listdir('data/blossom_cache')
    weather_files = os.listdir('data/weather_cache')

    #Only use files that we have both blossom and weather data for
    file_list = [file for file in blossom_files if file in weather_files]

    random.seed(8995)
    random.shuffle(file_list)

    #Split data into train, test, and validation sections with a [0.8, 0.1, 0.1] split

    trainval_split = int(0.8 * len(file_list))
    valtest_split = int(0.9 * len(file_list))

    splits = {
        "train": file_list[:trainval_split],
        "val": file_list[trainval_split:valtest_split],
        "test": file_list[valtest_split:]
    }
    #Store splits in a json for later use
    with open('data/splits.json', 'w') as fp:
        json.dump(splits, fp, indent=2)