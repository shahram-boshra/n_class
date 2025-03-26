import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    global_mean_pool,
    MessagePassing,
    GCNConv,
    GATConv,
    SAGEConv,
    GINConv,
    GraphConv,
    TransformerConv,
)

class CustomMPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return self.lin(x_j)

class MGModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        first_layer_type,
        second_layer_type,
        hidden_channels,
        dropout_rate,
        gat_heads = 1,
        transformer_heads = 1
    ):
        super(MGModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_channels = hidden_channels
        self.gat_heads = gat_heads
        self.transformer_heads = transformer_heads

        if first_layer_type == "custom_mp":
            self.conv1 = CustomMPLayer(in_channels, hidden_channels)
        elif first_layer_type == "gcn":
            self.conv1 = GCNConv(in_channels, hidden_channels)
        elif first_layer_type == "gat":
            self.conv1 = GATConv(in_channels, hidden_channels, heads=gat_heads, dropout=dropout_rate)
        elif first_layer_type == "sage":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
        elif first_layer_type == "gin":
            self.conv1 = GINConv(nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
        elif first_layer_type == "graph_conv":
            self.conv1 = GraphConv(in_channels, hidden_channels)
        elif first_layer_type == "transformer_conv":
            self.conv1 = TransformerConv(in_channels, hidden_channels, heads=transformer_heads)
        else:
            self.conv1 = GCNConv(in_channels, hidden_channels) # default case

        self.bcn1 = nn.BatchNorm1d(hidden_channels)

        if second_layer_type == "custom_mp":
            self.conv_gcn = CustomMPLayer(hidden_channels, hidden_channels)
        elif second_layer_type == "gcn":
            self.conv_gcn = GCNConv(hidden_channels, hidden_channels)
        elif second_layer_type == "gat":
            self.conv_gcn = GATConv(hidden_channels, hidden_channels, heads=gat_heads, dropout=dropout_rate)
        elif second_layer_type == "sage":
            self.conv_gcn = SAGEConv(hidden_channels, hidden_channels)
        elif second_layer_type == "gin":
            self.conv_gcn = GINConv(nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, hidden_channels)))
        elif second_layer_type == "graph_conv":
            self.conv_gcn = GraphConv(hidden_channels, hidden_channels)
        elif second_layer_type == "transformer_conv":
            self.conv_gcn = TransformerConv(hidden_channels, hidden_channels, heads=transformer_heads)
        else:
            self.conv_gcn = GCNConv(hidden_channels, hidden_channels) # Default case

        self.bcn_gcn = nn.BatchNorm1d(hidden_channels)
        self.conv2 = CustomMPLayer(hidden_channels, hidden_channels * 2)
        self.bcn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.linout = nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bcn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv_gcn(x, edge_index)
        x = self.bcn_gcn(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bcn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        #x = global_mean_pool(x, batch)
        x = self.linout(x)

        l1_norm = 0
        for param in self.parameters():
            l1_norm += torch.abs(param).sum()
        l1_reg = 0 # l1 lambda will be passed in from train loop

        return F.log_softmax(x, dim =1), l1_reg
