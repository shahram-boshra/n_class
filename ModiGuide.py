
target_csv # dataset.py
index_col   # dataset.py


target = torch.tensor(target, dtype = torch.float).reshape(1, -1) # target, dtype = torch.long) # data_loader.py


def forward(self, x, edge_index, batch): # (self, data) # for classification   # models.py
        #x, edge_index, batch = data.x, data.edge_index, data.batch
    .
    .
    . #       row, col = edge_index
    . #       edge_features = torch.cat([x[row], x[col]], dim = 1)
        
    . #       x = self.linear_out(edge_features)
    .
    .
    return x # F.log_softmax(x, dim =1)   # models.py


out_channels = dataset.target_df.shape[1]   # 2   # main.py


criterion = nn.HuberLoss(reduction = 'mean', delta = 0.1,) # nn.NLLLoss(reduction = 'mean')   #  main.py


out = model(batch.x, batch.edge_index, batch.batch) # model(batch)   #   training.py
target = batch.y # batch.edge_y for edge_level 
.
.
.
all_predictions.append(out.cpu().numpy())   # all_predictions.append(torch.argmax(out, dim=1).cpu().numpy())   #   training.py


# Note in Train, Valid, Test functions   num_graphs ---> num_nodes/num_edges

# delete global_mean_pool() in models.py for node and edge tasks
