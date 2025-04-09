import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import MessagePassing, GATv2Conv, global_mean_pool, GCNConv, SAGEConv




class MPNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNN, self).__init__(aggr='add')  
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.lin_update = torch.nn.Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index):
        # x: [N, in_channels], edge_index: [2, E]
        m = self.propagate(edge_index, x=x)  # Message passing
        x = self.lin_update(torch.cat([self.lin(x), m], dim=1))
        return F.leaky_relu(x, negative_slope=0.1)

    def message(self, x_j):
        # x_j: [E, in_channels]
        return self.lin(x_j)


class FIVEMPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(FIVEMPNN, self).__init__()
        self.conv1 = MPNN(in_channels, hidden_channels)
        self.conv2 = MPNN(hidden_channels, hidden_channels)
        self.conv3 = MPNN(hidden_channels, hidden_channels)  
        self.conv4 = MPNN(hidden_channels, hidden_channels)  
        self.conv5 = MPNN(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)  
        x = self.conv4(x, edge_index)  
        x = self.conv5(x, edge_index)
        x = torch_geometric.nn.global_mean_pool(x, batch)  # Global average pooling
        x = self.out(x)
        return F.log_softmax(x, dim=1)



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # First GCN layer
        self.conv1 = GCNConv(8, hidden_channels)
        # Second GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Third GCN layer
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # Fourth GCN layer
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        # Fifth GCN layer
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        # Fully connected layer
        self.out = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.conv4(x, edge_index)
        x = F.relu(x)

        x = self.conv5(x, edge_index)
        x = F.relu(x)

        # Global average pooling
        x = global_mean_pool(x, data.batch)
        
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)


class FiveLayerGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FiveLayerGAT, self).__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=8, concat=True)
        self.gat2 = GATv2Conv(8 * hidden_channels, hidden_channels, heads=8, concat=True)
        self.gat3 = GATv2Conv(8 * hidden_channels, hidden_channels, heads=8, concat=True)
        self.gat4 = GATv2Conv(8 * hidden_channels, hidden_channels, heads=8, concat=True)
        self.gat5 = GATv2Conv(8 * hidden_channels, hidden_channels, heads=8, concat=True)
        self.out = torch.nn.Linear(8 * hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)

        # Second GAT layer
        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # Third GAT layer
        x = self.gat3(x, edge_index)
        x = F.elu(x)

        # Fourth GAT layer
        x = self.gat4(x, edge_index)
        x = F.elu(x)

        # Fifth GAT layer
        x = self.gat5(x, edge_index)
        x = F.elu(x)

        # Global average pooling
        x = global_mean_pool(x, batch)

        # Output layer
        x = self.out(x)
        return F.log_softmax(x, dim=1)


class FiveLayerSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FiveLayerSAGE, self).__init__()
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)
        self.sage4 = SAGEConv(hidden_channels, hidden_channels)
        self.sage5 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # First SAGE layer
        x = self.sage1(x, edge_index)
        x = F.relu(x)

        # Second SAGE layer
        x = self.sage2(x, edge_index)
        x = F.relu(x)

        # Third SAGE layer
        x = self.sage3(x, edge_index)
        x = F.relu(x)

        # Fourth SAGE layer
        x = self.sage4(x, edge_index)
        x = F.relu(x)

        # Fifth SAGE layer
        x = self.sage5(x, edge_index)
        x = F.relu(x)

        # Global average pooling
        x = global_mean_pool(x, batch)

        # Output layer
        x = self.out(x)
        return F.log_softmax(x, dim=1)










class DumplingGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(DumplingGNN, self).__init__()
        self.mpnn1 = MPNN(8, hidden_channels)
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=8, concat=True)
        self.gat2 = GATv2Conv(8*hidden_channels, hidden_channels, heads=8, concat=True)
        self.gat3 = GATv2Conv(8*hidden_channels, hidden_channels, heads=8, concat=True)
        self.sage = SAGEConv(8*hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.mpnn1(x, edge_index)
        x = F.relu(x)

        x = self.gat1(x, edge_index)
        x = F.elu(x)


        x = self.gat2(x, edge_index)
        x = F.leaky_relu(x)

   
        x = self.gat3(x, edge_index)
        x = F.elu(x)


  
        x = self.sage(x, edge_index)
        x = F.relu(x)

  
        x = global_mean_pool(x, batch)

     
        x = self.out(x)
        return F.log_softmax(x, dim=1)

