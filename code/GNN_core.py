from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_add_pool
import torch
from torch_geometric.nn import GraphConv,TransformerConv,GCNConv
import random
import numpy as np
from os.path import exists
# import PDB2Graph
from torch_geometric.data import Data
# from proteingraph.pin import pdb2df
# from proteingraph import read_pdb
import GNN_core

from torch_geometric.nn import GATConv, GINConv
# from torch_geometric.nn.dense.linear import Linear

class GTN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers,heads):
        super(GTN, self).__init__()
        torch.manual_seed(12345)
        self.linear = Linear(num_node_features, num_node_features)
        self.bn0 = torch.nn.BatchNorm1d(num_node_features)
        
        self.conv1 = TransformerConv(num_node_features, hidden_channels, heads)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = TransformerConv(heads*hidden_channels, hidden_channels, heads)
        self.bn2 = torch.nn.BatchNorm1d(heads*hidden_channels)
        self.conv3 = TransformerConv(heads*hidden_channels, hidden_channels, heads)
        self.bn3 = torch.nn.BatchNorm1d(heads*hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(TransformerConv(heads*hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(heads*hidden_channels))

        self.lin = Linear(heads*hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.linear(x)
        x = self.bn0(x)
        x = x.relu()
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.lin(x)
        return x

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.linear = Linear(num_node_features, num_node_features)
        self.bn0 = torch.nn.BatchNorm1d(num_node_features)
        
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(GraphConv(hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.linear(x)
        x = self.bn0(x)
        x = x.relu()
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
                x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = self.lin(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.linear = Linear(num_node_features, num_node_features)
        self.bn0 = torch.nn.BatchNorm1d(num_node_features)
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(GCNConv(hidden_channels, hidden_channels))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))
       

        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
#     def forward(self, x, edge_index):
#         batch = torch.zeros((x.shape[0]), dtype=torch.int64)
#         print('batch',batch.shape)
#         print('Input',x.shape)
        # 1. Obtain node embeddings 
        x = x.float()
        x = self.linear(x)
        x = self.bn0(x)
        x = x.relu()
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
#         print('Step1:',x.shape)
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
                x = x.relu()
#         print('Step2:',x.shape)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
#         print('Step3:',x.shape)
        x = self.lin(x)

#         print('Output:',x.shape)
        
        return x
    
class GAT_jz(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers,heads):
        super(GAT_jz, self).__init__()
        torch.manual_seed(12345)
#         self.linear = torch_geometric.nn.dense.linear.Linear(num_node_features, num_node_features)
        self.linear = Linear(num_node_features, num_node_features)
        self.bn0 = torch.nn.BatchNorm1d(num_node_features)
        
        # self.linear01 = Linear(num_node_features, num_node_features)
        # self.bn01 = torch.nn.BatchNorm1d(num_node_features)
        
        self.conv1 = GATConv(num_node_features, hidden_channels, heads)
        self.bn1 = torch.nn.BatchNorm1d(heads*hidden_channels)
        self.conv2 = GATConv(heads*hidden_channels, hidden_channels, heads)
        self.bn2 = torch.nn.BatchNorm1d(heads*hidden_channels)
        self.conv3 = GATConv(heads*hidden_channels, hidden_channels, heads)
        self.bn3 = torch.nn.BatchNorm1d(heads*hidden_channels)

        self.conv=torch.nn.ModuleList()
        self.bn=torch.nn.ModuleList()
        for l in range(int(num_layers)):
            self.conv.append(GATConv(heads*hidden_channels, hidden_channels, heads))
            self.bn.append(torch.nn.BatchNorm1d(heads*hidden_channels))
        
        # self.linear10 = Linear(heads*hidden_channels, heads*hidden_channels)
        # self.bn10 = torch.nn.BatchNorm1d(heads*hidden_channels)
        
        self.lin = Linear(heads*hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
#         print('x:',x.shape)
        x = x.float()
        x = self.linear(x)
        x = self.bn0(x)
        x = x.relu()
        
        # x = self.linear01(x)
        # x = self.bn01(x)
        # x = x.relu()
        
        # print('x1.dtype: ',x.dtype)
        x = self.conv1(x, edge_index)
        x = self.bn1(x.float())
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x.float())
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.bn3(x.float())
        x = x.relu()
        # 1.1 Additional deep layers
        if len(self.conv) > 0:
            for index,conv_i in enumerate(self.conv):
                x = conv_i(x,edge_index)
                x = self.bn[index](x)
                x = x.relu()
        
        # x = self.linear10(x)
        # x = self.bn10(x)
        # x = x.relu()
        
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        
        x = self.lin(x)
        return x

class GIN(torch.nn.Module):
    def __init__(self, hidden_channels,num_node_features,num_classes,num_layers):
        super(GIN, self).__init__()
        torch.manual_seed(12345)
        self.linear = Linear(num_node_features, num_node_features)
        self.bn0 = torch.nn.BatchNorm1d(num_node_features)
        
        self.conv1 = GINConv(
            Sequential(Linear(num_node_features, hidden_channels),
                       BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), BatchNorm1d(hidden_channels), ReLU(),
                       Linear(hidden_channels, hidden_channels), ReLU()))

        self.lin1 = Linear(hidden_channels*3, hidden_channels*3)
        self.lin2 = Linear(hidden_channels*3, num_classes)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.linear(x)
        x = self.bn0(x)
        x = x.relu()

        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        h3 = global_add_pool(h3, batch)
        
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)

        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        x = self.lin2(h)

        return x
    
### helper function to output the internal activations
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def train(model,train_loader,optimizer,criterion):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        #model.conv1.register_forward_hook(get_activation('conv3'))
        out = model.forward(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
#         out = model(data.x, data.edge_index)  # Perform a single forward pass.
        
        targets = data.y
#         targets = torch.unsqueeze(data.y, 1)
        loss = criterion(out, targets)  # Compute the loss.
#         print('Output:',out.shape)
#         print('Output[:2,:]:',out[:2,:])
#         print('Label:', data.y.shape)
#         print('loss',loss)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model,loader):
    model.eval()
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        # print(out.shape)
        # print(out[:5])
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def predict(model,loader):
    model.eval()
    pred=[]
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred.append(out.argmax(dim=1).tolist())  # Use the class with highest probability.
          # Check against ground-truth labels.
    return pred  # Derive ratio of correct predictions.

def loss(model,loader,criterion):
    model.eval()
    loss=0.
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        loss += criterion(out, data.y).sum()
    return loss/len(loader.dataset)

def get_info_dataset(dataset, verbose=False):
    """Determines the number of inputs labeled one and zero in a dataset."""
    zeros = 0
    ones = 0
    for data in dataset:
        label = data.y.item()
        if label == 0:
            zeros+=1
        elif label ==1:
            ones+=1
    if verbose:
        print(f'In this dataset, there are {zeros} inputs labeled "0" and {ones} inputs labeled "1".')
    return (ones, zeros)

def balance_dataset(dataset):
    ones, zeros = get_info_dataset(dataset)
    if zeros==ones:
        return dataset
    if zeros>ones:
        major=zeros
        minor=ones
        the_major_one=0
    else:
        major=ones
        minor=zeros
        the_major_one=1

    ratio = float((major-minor)/(major))
    balanced = []
    for item in dataset:
        label = item.y.item()
        if label == the_major_one:
            if random.random()>ratio:
                balanced.append(item)
        else:
            balanced.append(item)
    return balanced


def convert_pdb2graph(input):
 
    pdb_path=input[0]
    my_protein=input[1]
    featureData=input[2]
    graph_labels=input[3]
    protein_index=input[4]
    type_input=input[5]
    if type_input=='AlphaFold':
        path=str(pdb_path)+'/'+'AF-'+str(my_protein)+'-F1-model_v3.pdb'
    if type_input=='PDB':
        path=str(pdb_path)+'/'+str(my_protein)+'.pdb'
    else:
        print('Unkown input type, must be AlphaFold or PDB')
        return 
    if exists(path):
        try:
            pdb_df=pdb2df(path)
            res2node=PDB2Graph.residueID2nodeID(pdb_df)
            #G=read_pdb(str(pdb_path)+'/'+str(my_protein)+'.pdb')
            if type_input=='AlphaFold':
                G=read_pdb(str(pdb_path)+'/'+'AF-'+str(my_protein)+'-F1-model_v3.pdb')
            if type_input=='PDB':
                G=read_pdb(str(pdb_path)+'/'+str(my_protein)+'.pdb')
            else:
                print('Unkown input type, must be AlphaFold or PDB')
                return
            PDB2Graph.add_pi_pi_interactions(G,pdb_df)

            node=PDB2Graph.node(G,res2node)
            edge=PDB2Graph.edge(G,res2node)
            print("Loaded ",str(my_protein))
        except (IndexError, ValueError):
            print("Can't load ",str(my_protein))
            return

        
    ### readin feature list of all amino acids
        try:
            complete_list_feature=[]
            for aminoAcid in featureData.buildProtein(node):
                my_features=[float(tmp) for tmp in list(aminoAcid)]
                complete_list_feature.append(my_features)

            nodes_features=torch.tensor(complete_list_feature,dtype=torch.float) # feature vector of all nodes
            edge_index = torch.tensor(edge, dtype=torch.long) # edges, 1st list: index of the source nodes, 2nd list: index of target nodes.
            my_label=torch.tensor(graph_labels[protein_index], dtype=torch.long)
            g = Data(x=nodes_features, edge_index=edge_index,y=my_label)

            return g
        except KeyError:
            print("Failed loading aminoacids info for ",str(my_protein))
            return

