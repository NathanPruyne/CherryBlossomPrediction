import torch

#Finds the average distance between the predicted day and the ground truth's day, in days
def distance_to_correct(predictions: torch.Tensor, labels: torch.Tensor):
    distances = []
    for prediction, label in zip(predictions, labels): #Iterate over batch
        prediction_max = torch.argmax(prediction)
        label_max = torch.argmax(label)
        distances.append(torch.abs(prediction_max - label_max))

    return sum(distances) / len(distances)