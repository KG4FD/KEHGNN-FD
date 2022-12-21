import torch
import torch_geometric.transforms as T
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, GATv2Conv, Linear
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_curve
from torch.nn import ReLU, Sigmoid
import matplotlib.pyplot as plt
import sklearn.metrics as sm


class KGHeteroGNN_Initial(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels, add_self_loops=False),
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
            if i!= num_layers-1:
                hidden_channels = int(hidden_channels/2)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
           
        return self.lin(x_dict['news'])
      

class KGHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels),
                ('entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels),
                ('news', 'on', 'topic'): GATv2Conv((-1, -1), hidden_channels),
                ('topic', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels),
                ('news', 'has', 'kg_entities'): GATv2Conv((-1, -1), hidden_channels),
                ('kg_entities', 'in', 'news'): GATv2Conv((-1, -1), hidden_channels),

                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            if i!= num_layers-1:
                hidden_channels = int(hidden_channels/2)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            
        return self.lin(x_dict['news'])

class KGHeteroGNN1(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                ('news', 'has', 'entities'): GATv2Conv((-1, -1), hidden_channels),
                ('entities', 'in', 'news'): GATConv((-1, -1), hidden_channels),
                ('news', 'on', 'topic'): GATConv((-1, -1), hidden_channels),
                ('topic', 'in', 'news'): GATConv((-1, -1), hidden_channels),
                ('entities', 'similar', 'entities'): GATv2Conv(-1, hidden_channels),
            }, aggr='sum')
            self.convs.append(conv)
            if i!= num_layers-1:
                hidden_channels = int(hidden_channels/2)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return self.lin(x_dict['news'])

def train(model, data, args):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data['news'].train_mask
        #print(out[mask].shape, data['news'].y[mask])
        loss = criterion(out[mask], data['news'].y[mask])
        loss.backward()
        optimizer.step()
    
    pred = out[data['news'].train_mask].argmax(dim=1).cpu()
    y = data['news'].y[[data['news'].train_mask]].cpu()

    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, )
    f1 = f1_score(y, pred)
    recall = recall_score(y, pred, )
    print(f"Training Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},F1: {f1:.4f}")
    with open("./Generalization.result", "a", encoding="utf8") as f:
        f.write(f"Dataset: {args.dataset}; Train_Acc: {acc:.4f}; Train_Precision: {precision:.4f}; Train_Recall: {recall:.4f}; Train_F1: {f1:.4f} \n")

def test(model, data, args, i):
    out = model(data.x_dict, data.edge_index_dict)
    pred = out[data['news'].test_mask].argmax(dim=1).cpu()
    print(out[data['news'].test_mask])

    y = data['news'].y[[data['news'].test_mask]].cpu()
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred, )
    f1 = f1_score(y, pred)
    recall = recall_score(y, pred, )
 
    print(f"Testing Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},F1: {f1:.4f}")
    with open("./Generalization.result", "a", encoding="utf8") as f:
        f.write(f"Dataset: {args.dataset}; Test_Acc: {acc:.4f}; Test_Precision: {precision:.4f}; Test_Recall: {recall:.4f}; Test_F1: {f1:.4f} \n")
    

    # draw ROC curve 
    fpr, tpr, threshold = roc_curve(y, pred)
    roc_auc = sm.auc(fpr, tpr)

    plt.figure()
    lw = 2
    
    plt.figure(figsize=(10, 10))
    # fake postive is x-axis， true positive is y-axis
    plt.plot(fpr, tpr, color='red', lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


    plt.title('Receiver operating characteristic curve(ROC)_{}'.format(args.dataset))
    plt.legend(loc="lower right")
    plt.show()
    
