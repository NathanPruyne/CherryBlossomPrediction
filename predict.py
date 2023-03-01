import argparse
from pathlib import Path
import json
import os

from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

from dataset import BlossomDataset
from blossom_model import Convolution
from metrics import distance_to_correct

def predict(model_file, output_dir, dataset=None, test_set=False, final_format=False):
    if not dataset and not test_set:
        raise ValueError("One of dataset or test_set must be used")
    inputs = []
    truths = []
    files = []
    if test_set: #Get the test set from the splits
        with open('data/splits.json', 'r') as fp:
            splits = json.load(fp)
        dataset = BlossomDataset(splits['test'])
        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        for (blossom, weather, file) in test_loader:
            inputs.append(weather)
            truths.append(blossom)
            files.append(file[0])
    else:
        files = [f for f in os.listdir(dataset) if os.path.splitext(f)[1] == '.npy']
        for file in files:
            tensor = torch.Tensor(np.load(os.path.join(dataset, file))).unsqueeze(0) #Add a "batch dimension"
            inputs.append(tensor)
    
    with torch.no_grad():
    
        model = Convolution()
        model.load_state_dict(torch.load(model_file))
        model.eval()

        outputs = []
        for input in tqdm(inputs, desc="Predicting: "):
            outputs.append(model(input).squeeze(0)) #Remove batch dimension

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        if final_format:
            years = np.arange(2023, 2033)
            final = np.zeros((10, 5))
            locations = ["year", "kyoto", "liestal", "washingtondc", "vancouver"]
            #Populate in specific order designated
            for r, year in enumerate(years):
                for c, loc in enumerate(locations):
                    if loc == "year":
                        datapoint = year
                    else:
                        this_output = outputs[files.index(loc + "_" + str(year) + ".npy")]
                        datapoint = np.argmax(this_output)
                    final[r][c] = datapoint
            np.savetxt(os.path.join(output_dir, "results.csv"), final, delimiter=',', fmt='%d', header='"year","kyoto","liestal","washingtondc","vancouver"', comments='')
        else:
            for file, output in zip(files, outputs):
                predicted_day = np.argmax(output)
                np.savetxt(os.path.join(output_dir, file), predicted_day.unsqueeze(0), fmt='%d')
        
        if truths: #This was the test set, we can get some data out
            distances = []
            for output, truth in zip(outputs, truths):
                distances.append(distance_to_correct(output, truth))
            print(f"Test set distance: {sum(distances) / len(distances)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_set', action='store_true', help='If used, will evaluate on the test set')
    parser.add_argument('--dataset', type=Path, help="If not test_set, the location of a folder with data to predict")
    parser.add_argument('--model', type=Path, help="Location of model file")
    parser.add_argument('--output', type=Path, help="Final to output to")
    parser.add_argument('--final_format', action='store_true', help='If used, will output in the format needed for the final output')

    args = parser.parse_args()

    predict(args.model, args.output, args.dataset, args.test_set, args.final_format)