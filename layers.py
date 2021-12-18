import torch
import torch.nn as nn
import math

from torch.nn import Linear, LSTM
from torch.nn import LayerNorm, BatchNorm1d
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNConv(nn.Module):
    def __init__(self, task, conv_name, in_channels, out_channels, norm,
                 n_heads=[1, 1], iscat=[False, False], dropout_att=0.):
        super(GNNConv, self).__init__()
        self.task = task

        if conv_name == 'gcn_conv':
            self.conv  = GCNConv(in_channels, out_channels)
            self.conv_ = self.conv.lin
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.linear = nn.Linear(in_channels, out_channels)

        elif conv_name == 'sage_conv':
            self.conv  = SAGEConv(in_channels, out_channels)
            self.conv_ = [self.conv.lin_r, self.conv.lin_l]
            if(self.task == 'inductive'):  # if transductive, we dont use linear
                self.linear = nn.Linear(in_channels, out_channels)

        elif conv_name == 'gat_conv':
            if iscat[0]: # if previous gatconv's cat is True
                in_channels = in_channels * n_heads[0]
            self.conv  = GATConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 heads=n_heads[1],
                                 concat=iscat[1],
                                 dropout=dropout_att)
            self.conv_ = self.conv.lin_src
            if iscat[1]: # if this gatconv's cat is True
                out_channels = out_channels * n_heads[1]

            if self.task == 'inductive':  # if transductive, we dont use linear
                self.linear = nn.Linear(in_channels, out_channels)
        
        if norm == 'LayerNorm':
            self.norm = LayerNorm(out_channels)
        elif norm == 'BatchNorm1d':
            self.norm = BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()


    def forward(self, x, edge_index):
        if self.task == 'transductive':
            x = self.conv(x, edge_index)
        elif self.task == 'inductive':
            x = self.conv(x, edge_index) + self.linear(x)

        return self.norm(x)



# if cfg.skip_connection is summarize
class SummarizeSkipConnection(nn.Module):
    def __init__(self, summary_mode, att_mode, channels, num_layers):
        super(SummarizeSkipConnection, self).__init__()
        self.summary_mode = summary_mode
        self.att_mode = att_mode
        
        if self.summary_mode == 'lstm':
            out_channels = (num_layers * channels) // 2
        else: # if self.summary_mode == 'vanilla' or 'roll'
            out_channels = channels
        
        self.lstm = LSTM(channels, out_channels,
                             bidirectional=True, batch_first=True)
        self.att = Linear(2 * out_channels, 1)
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.kldiv = nn.KLDivLoss(reduction='none')

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.att.reset_parameters()

    def forward(self, hs):
        h = torch.stack(hs, dim=1)  # h is (n, L, d).

        # 'Summary' takes h as input, query and key vector as output
        if self.summary_mode == 'vanilla':
            query = h.clone() # query's l-th row is h_i^l
            n_layers = h.size()[1]
            key = query[:, -1, :].repeat(n_layers, 1, 1).permute(1,0,2) # key's all row is h_i^L

        elif self.summary_mode == 'roll':
            query = h.clone() # query's l-th row is h_i^l
            key = torch.roll(h.clone(), -1, dims=1) # key's l-th row is h_i^(l+1)
            query, key, h = query[:, :-1, :], key[:, :-1, :], h[:, :-1, :]

        elif self.summary_mode == 'lstm':
            alpha, _ = self.lstm(h) # alpha (n, L, dL). dL/2 is hid_channels of forward or backward LSTM
            out_channels = alpha.size()[-1]
            query, key = alpha[:, :, :out_channels//2], alpha[:, :, out_channels//2:]

        # 'Attention' takes query and key as input, alpha as output
        if self.att_mode == 'dp':
            alpha = (query * key).sum(dim=-1) / math.sqrt(query.size()[-1])

        elif self.att_mode == 'ad':
            query_key = torch.cat([query, key], dim=-1)
            alpha = self.att(query_key).squeeze(-1)
            
        elif self.att_mode == 'mx': 
            query_key = torch.cat([query, key], dim=-1)
            alpha_ad = self.att(query_key).squeeze(-1)
            alpha = alpha_ad * torch.sigmoid((query * key).sum(dim=-1))

        alpha_softmax = torch.softmax(alpha, dim=-1)
        return (h * alpha_softmax.unsqueeze(-1)).sum(dim=1) # h_i = \sum_l alpha_i^l * h_i^l


# if cfg.skip_connection is in [vanilla, res, dense, highway]
class SkipConnection(nn.Module):
    def __init__(self, skip_connection, n_hidden):
        super(SkipConnection, self).__init__()
        self.skip_connection = skip_connection
        if self.skip_connection == 'highway':
            self.linear = Linear(n_hidden, n_hidden)

    def forward(self, h, x):
        if self.skip_connection == 'vanilla':
            return h

        elif self.skip_connection == 'res':
            return h + x

        elif self.skip_connection == 'dense':
            return torch.cat([h, x], dim=-1)
            
        elif self.skip_connection == 'highway':
            gating_weights = torch.sigmoid(self.linear(x))
            ones = torch.ones_like(gating_weights)
            return h*gating_weights + x*(ones-gating_weights) # h*W + x*(1-W)
