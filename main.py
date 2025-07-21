import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])

y_train = torch.tensor([0, 0, 0, 1, 1])


X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])

y_test = torch.tensor([0, 1])

class ToyDataset(Dataset):
    def __init__(self, X, y): #we initialize the dataset using an array of input data, "X", and an array of output answers, "y"
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x,one_y # returns the tensor and answer on position "index" in the arrays we passed to this dataset
    
    def __len__(self): #might be used by the DataLoader in order to know how long to iterate over the dataset
        return self.labels.shape[0]
    
train_ds = ToyDataset(X_train, y_train)

test_ds = ToyDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0
)