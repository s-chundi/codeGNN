import torch_geometric as pyg
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import dataset
import torch
import torch.optim as optim
from tqdm import trange
import copy
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

args = {'model_type': 'GS', 'num_layers': 5, 'heads': 1, 'batch_size': 3, 'embedding_dim': 210, 'hidden_dim': 210, 'dropout': 0.5, 
            'epochs': 10, 'opt': 'adam', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01}

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        for l in range(args.num_layers-1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout), 
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers

    def build_conv_model(self, model_type):
        if model_type == 'GS':
            return GraphSage
        if model_type == 'GAT':
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
          
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,training=self.training)

        x = self.post_mp(x)

        return x
    
    def loss(self, predictions, label):
#         predictions = F.normalize(predictions, dim=1)
#         sim = ((predictions[0] @ predictions[1]).unsqueeze(0) + 1) / 2 - 1e-4
#         dif = ((predictions[1] @ predictions[2]).unsqueeze(0) + 1) / 2 + 1e-4
#         tru = label[0] == label[1]
#         otru = label[1] == label[2]
#         tru = tru.type(torch.float32).unsqueeze(0)
#         otru = otru.type(torch.float32).unsqueeze(0)
#         try:
#             loss = nn.BCELoss()(sim, tru) + nn.BCELoss()(dif, otru)
#         except:
#             print(sim)
#             print(tru)
#         return loss
        
        x = F.normalize(predictions, dim = 1)
        y = label
        similarity_matrix = x @ x.T
        label_matrix = torch.eq(y.view(-1, 1), y.view(1, -1))

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        #flatten
        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        sp = similarity_matrix[positive_matrix] 
        sn = similarity_matrix[negative_matrix]
   
        alpha_p = torch.relu(- sp.detach() + 1 + self.margin)
        alpha_n = torch.relu(sn.detach() + self.margin)

        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p = - alpha_p * (sp - delta_p) * self.gamma
        logit_n = alpha_n * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim = 0) + torch.logsumexp(logit_p, dim = 0))
        # loss = nn.CrossEntropyLoss()
        print(loss)
        return loss
    
class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class GAT(MessagePassing):
    # TODO: there might be some bug with heads
    def __init__(self, in_channels, out_channels, heads = 1,
                 negative_slope = 0.2, dropout = 0.2, **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin_l = nn.Linear(in_channels, out_channels * heads)
        self.lin_r = nn.Linear(in_channels, out_channels * heads)
        self.att_l = nn.Parameter(torch.rand(out_channels, heads))
        self.att_r = nn.Parameter(torch.rand(out_channels, heads))
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels
        # print(x.shape)
        # print(f'edge index shape {edge_index.shape}')
        N = x.shape[0]
        x = x.type(torch.float32)
        # print(N, H, C)
        # print(self.lin_l.weight.dtype)
        # print(x.shape, self.lin_l.weight.shape)
        # x = torch.stack([x] * self.heads)
        source = self.lin_l(x)
        source = torch.reshape(source, (N, H, C))
        target = self.lin_r(x)
        target = torch.reshape(target, (N, H, C))
        alpha_l = torch.matmul(source, self.att_l)
        alpha_r = torch.matmul(target, self.att_r)
        out = self.propagate(edge_index, x = (source, target), alpha = (alpha_l, alpha_r), size=size)
        out = torch.reshape(out, (N, H*C))
        # print(out.shape, '\n\n\n\n\n')
        return out


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        
        E = x_j.shape[0]
        aw = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)
        alphas = softmax(aw, index = index, ptr = ptr, num_nodes = size_i)
        alphas = F.dropout(alphas, self.dropout)
        out = alphas @ x_j

        return out


    def aggregate(self, inputs, index, dim_size = None):
        out = torch_scatter.scatter(inputs, index, dim = 0, dim_size=dim_size, reduce='sum')

        return out
    
class GraphSage(MessagePassing):
    
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size = None):
        out = self.propagate(edge_index, x=(x, x), size=size).type(torch.float32)
        out = self.lin_r(out)
        
        x = x.type(torch.float32)
        out += self.lin_l(x)
     

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j):
        out = x_j

        return out

    def aggregate(self, inputs, index, dim_size = None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim=node_dim, dim_size=dim_size, reduce='mean')

        return out
    
def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    return optimizer

def train(train_loader, valid_loader, args, model = None):
    # test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # build model
    if model == None:
        model = GNNStack(train_loader.dataset.num_node_features, args.hidden_dim, args.embedding_dim, args)
    opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    valid_errs = []
    best_err = float('inf')
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            pred = model(batch)
            predictions = torch_scatter.scatter(pred, batch.batch, dim = 0, reduce='mean')
            label = batch.task_label
            loss = model.loss(predictions, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
          valid_err = test(valid_loader, model)
          valid_errs.append(valid_err)
          if valid_err < best_err:
            best_err = valid_err
            best_model = copy.deepcopy(model)
        else:
          valid_errs.append(valid_errs[-1])
    
    return valid_errs, losses, best_model, best_err

def test(loader, test_model):
    test_model.eval()

    error = 0
    for batch in loader:
        with torch.no_grad():
            pred = test_model(batch)
            predictions = torch_scatter.scatter(pred, batch.batch, dim = 0, reduce='mean')
            label = batch.task_label
            loss = test_model.loss(predictions, label)
            
        error += loss
    # print(f'Total error: {error}')
    return error

if __name__ == '__main__':
    args = objectview(args)
    # ds = dataset.txt_to_pyg_data('simple.txt', task_label=1, index=7)
    # ds2 = dataset.txt_to_pyg_data('simple.txt', task_label=1, index=4)
    # ds3 = dataset.txt_to_pyg_data('input.txt', task_label=2, index=3)
    # print(ds.x.shape)
    # bds = Batch.from_data_list([ds, ds2, ds3])
    # print(type(bds))

    train_loader = DataLoader(torch.load('data/poj-104/train.pt'))
    valid_loader = DataLoader(torch.load('data/poj-104/valid.pt'))
    test_loader = DataLoader(torch.load('data/poj-104/test.pt'))

    valid_errs, losses, best_model, best_err = train(train_loader, valid_loader, args)

    print("Minimum loss: {0}".format(min(losses)))

    # Run test for our best model to save the predictions!
    test(test_loader, best_model)
