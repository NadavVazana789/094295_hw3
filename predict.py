from torch_geometric.nn import Sequential, GCNConv
from torch.nn import Linear, ReLU
from dataset import HW3Dataset
import torch
import pandas as pd
MODEL_PATH = 'model.pt'



def main():
    model = Sequential('x, edge_index', [
    (GCNConv(128, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    (GCNConv(64, 64), 'x, edge_index -> x'),
    ReLU(inplace=True),
    Linear(64, 40),])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    data = HW3Dataset(root='data/hw3/')[0]
    pred = torch.argmax(model(data.x, data.edge_index), dim=1)
    pd.DataFrame({'idx': range(0, len(pred)), 'prediction': pred}).to_csv('prediction.csv')

if __name__ == '__main__':
    main()