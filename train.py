import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import argparse
import json
import time

from dataset import BlossomDataset
from blossom_model import Convolution
from metrics import distance_to_correct

def train(epochs, model_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('data/splits.json', 'r') as fp:
        splits = json.load(fp)

    #Generate datasets with files from the splits
    train_dataset = BlossomDataset(splits['train'])
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    val_dataset = BlossomDataset(splits['val'])
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=True)

    model = Convolution().to(device=device)
    model.train()

    loss_func = torch.nn.MSELoss() #Use a mean squared error loss

    optimizer = torch.optim.Adam(model.parameters())

    previous_losses = []
    best_distance = float('inf')

    start = time.time()

    for epoch in range(0, epochs):
        epoch_loss = []
        epoch_distance = []

        for (blossom, weather, _) in tqdm(train_loader, desc='Training epoch ' + str(epoch)):
            optimizer.zero_grad()

            blossom, weather = blossom.to(device), weather.to(device)
            blossom = torch.unsqueeze(blossom, 1) #Add channels dimension

            prediction = model(weather)

            loss = loss_func(prediction, blossom)
            epoch_loss.append(loss.item())

            epoch_distance.append(distance_to_correct(prediction, blossom))

            loss.backward()

            optimizer.step()

        #Track loss and distance for output
        print(f"Train loss: {sum(epoch_loss) / len(epoch_loss)}")

        print(f"Train distance: {sum(epoch_distance) / len(epoch_distance)}")

        val_loss = []
        val_distance = []
        #Check on validation set
        for (blossom, weather, _) in tqdm(val_loader, desc='Validating epoch ' + str(epoch)):
            blossom, weather = blossom.to(device), weather.to(device)
            blossom = torch.unsqueeze(blossom, 1)

            prediction = model(weather)

            loss = loss_func(prediction, blossom)

            val_loss.append(loss.item())

            val_distance.append(distance_to_correct(prediction, blossom))
        
        val_distance_avg = sum(val_distance) / len(val_distance)

        print(f"Validation loss: {sum(val_loss) / len(val_loss)}")

        print(f"Validation distance: {val_distance_avg}")

        previous_losses.append(val_distance_avg) #Track 100 last loss values

        if len(previous_losses) > 100: previous_losses.pop(0)

        #Only save model if we have found a new optimum
        if val_distance_avg < best_distance:
            torch.save(model.state_dict(), model_file)
            print(f"Validation distance improved to {val_distance_avg}, model saved")
            best_distance = val_distance_avg

        #Early stopping: if we detect that the validation distance is no longer improving, this may signal overfittng to the train data, so training should be halted
        if epoch > 100 and min(previous_losses) == previous_losses[0]: #Most recent was the worst in a while
            print(f"Val distance no longer improving, stopping early")
            break


    print(f"Final best validation distance is {best_distance}")
    print(f"Training took {time.time() - start} seconds")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='Where to store models')
    parser.add_argument('--epochs', help='Number of epochs')
    
    args = parser.parse_args()

    train(int(args.epochs), args.model_file)
