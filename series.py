import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class SeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        x = self.data[index:index+self.seq_len]
        y = self.data[index+self.seq_len:index+self.seq_len+self.pred_len]

        x_tensor = torch.FloatTensor(x).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)

        return x_tensor, y_tensor
    

def SeriesSetBuilder(dataset, num_test_labels, seq_len, pred_len):
    datalen = len(dataset)
    
    if datalen < seq_len + num_test_labels:
        raise ValueError("Sequence Length + # of testing labels cannot be greater than Data set length. Try Again.")
    
    normed_dataset = (dataset-np.mean(dataset))/np.std(dataset)

    test_dataset = normed_dataset[datalen-seq_len-num_test_labels:]
    test_set = SeriesDataset(test_dataset,seq_len,pred_len)

    train_dataset = normed_dataset[:-num_test_labels]
    print(len(train_dataset))
    train_set = SeriesDataset(train_dataset,seq_len,pred_len)

    return train_set, test_set


def GetLabels(dataset, train_set, test_set):
    train_labels = []
    test_labels = []
    
    for i in range(len(train_set)):
        _, label = train_set[i]
        train_labels.append(label)
    
    for i in range(len(test_set)):
        _, label = test_set[i]
        test_labels.append(label)

    return train_labels, test_labels


def SplitDataPlotter(dataset, train_set, test_set):
    datalen = len(dataset)
    train_labels, test_labels = GetLabels(dataset,train_set,test_set)
  

    plt.title(
        'Label Visualization', 
        loc = 'left', 
        pad = 10, 
        fontdict = {
            'size': 14, 'color': 
            'teal', 
            'weight': 'bold'
        }
    )
    
    plt.plot((dataset-np.mean(dataset))/np.std(dataset), color = "blue", label="Original Data")

    train_x = np.array(range(len(train_labels)))+train_set.seq_len
    plt.plot(train_x,train_labels, color="green", label="Train Labels")

    test_x = np.array(range(len(test_labels)))+datalen-len(test_labels)
    plt.plot(test_x,test_labels,color="orange",label="Test Labels")

    plt.legend()
    plt.show()


def SeriesModelTrainer(model, train_loader, loss_fn, optimizer):
    batch_size = train_loader.batch_size
    size = len(train_loader.dataset)


    model.train()

    for batch, (X,y) in enumerate(train_loader):
        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.item(), batch*batch_size
            print(f"Loss: {loss:>7f}    [{current:<5d}/{size:>5d}]")

def SeriesModelTester(model, test_loader, loss_fn, final = False):
    num_batches = len(test_loader) 
    
    model.eval()
    
    preds = []
    test_loss = 0
    with torch.no_grad():
        for X,y in test_loader:
            pred = model(X)
            preds.append(pred)
            test_loss += loss_fn(pred,y).item()
    
    test_loss /= num_batches
    print(f"Test Error: \n Avg Loss: {test_loss:>8f}")

    if final:
        return np.array(preds).flatten()

def SeriesTrainNTest(
        model, 
        train_loader, test_loader,
        loss_fn, optimizer
        ):

    SeriesModelTrainer(model,train_loader,loss_fn,optimizer)
    SeriesModelTester(model,test_loader,loss_fn)

def SeriesPredPlotter(dataset, train_set, test_set, predictions):
    datalen = len(dataset)
    _, test_labels = GetLabels(dataset,train_set,test_set)
  

    plt.title(
        'Predictions vs Labels', 
        loc = 'left', 
        pad = 10, 
        fontdict = {
            'size': 14, 'color': 
            'teal', 
            'weight': 'bold'
        }
    )
    
    plt.plot((dataset-np.mean(dataset))/np.std(dataset), color = "blue", label="Original Data")

    test_x = np.array(range(len(test_labels)))+datalen-len(test_labels)
    plt.plot(test_x,predictions, color="red", label="Predictions")
    plt.plot(test_x,test_labels,color="green",label="Test Labels")

    plt.legend()
    plt.show()




        
