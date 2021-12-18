import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GNNConv, SummarizeSkipConnection, SkipConnection


def conv_for_gpumemory(x_all, loader, conv, device):
    xs = []
    for batch_size, n_id, adj in loader:
        edge_index, _, size = adj.to(device)
        x = x_all[n_id].to(device)
        x_target = x[:size[1]]
        x = conv((x, x_target), edge_index)
        xs.append(x)
    x_all = torch.cat(xs, dim=0)
    return x_all


class SummarizeSAGE(nn.Module):
    def __init__(self, cfg):
        super(SummarizeSAGE, self).__init__()
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer

        self.convs = nn.ModuleList()
        self.convs.append(GNNConv(cfg.task, 'sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm))
        for _ in range(1, cfg.n_layer):
            self.convs.append(GNNConv(cfg.task, 'sage_conv', cfg.n_hid, cfg.n_hid, cfg.norm))

        self.attention_skip = SummarizeSkipConnection(summary_mode = cfg.summary_mode,
                                                      att_mode     = cfg.att_mode, 
                                                      channels     = cfg.n_hid, 
                                                      num_layers   = cfg.n_layer)
        self.out_lin = nn.Linear(cfg.n_hid, cfg.n_class)

    def forward(self, x, adjs):
        xs = []
        for l, (edge_index, _, size) in enumerate(adjs): # size is [B_l's size, B_(l+1)'s size]
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[l]((x, x_target), edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        batch_size = xs[-1].size()[0]
        xs = [x[:batch_size] for x in xs]

        h = self.attention_skip(xs) # xs = [h^1,h^2,...,h^L], each h^l is (n, d)
        return self.out_lin(h)

    def inference(self, x, loader, device):
        xs = []
        for conv in self.convs:
            x = conv_for_gpumemory(x, loader, conv, device)
            x = F.relu(x)
            xs.append(x)

        h = self.attention_skip(xs)
        return self.out_lin(h)


class SAGE(nn.Module):
    def __init__(self, cfg):
        super(SAGE, self).__init__()
        self.dropout = cfg.dropout
        self.n_layer = cfg.n_layer

        self.in_conv = GNNConv(cfg.task, 'sage_conv', cfg.n_feat, cfg.n_hid, cfg.norm)

        self.mid_convs = nn.ModuleList()
        self.skips = nn.ModuleList()
        for l in range(1, cfg.n_layer-1):
            if cfg.skip_connection != 'dense':
                in_channels = cfg.n_hid
            else: # if skip connection is dense
                in_channels = cfg.n_hid*l
            self.mid_convs.append(GNNConv(cfg.task, 'sage_conv', in_channels, cfg.n_hid, cfg.norm))
            self.skips.append(SkipConnection(cfg.skip_connection, cfg.n_hid))

        if cfg.skip_connection != 'dense':
            in_channels = cfg.n_hid
        else: # if skip connection is dense
            in_channels = cfg.n_hid*(cfg.n_layer-1)
        self.out_conv = GNNConv(cfg.task, 'sage_conv', in_channels, cfg.n_class, norm='None')

    def forward(self, x, adjs):
        in_adj, mid_adjs, out_adj = adjs[0], adjs[1:-1], adjs[-1]

        x_target = x[:in_adj.size[1]] # Target nodes are always placed first.
        x = self.in_conv((x, x_target), in_adj.edge_index) # x's shape is (B_l's size, hid) -> (B_(l+1)'s size, hid)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)

        for mid_adj, mid_conv, skip in zip(mid_adjs, self.mid_convs, self.skips): # size is [B_l's size, B_(l+1)'s size]
            x_target = x[:mid_adj.size[1]]
            h = mid_conv((x, x_target), mid_adj.edge_index)
            h = F.relu(h)
            x = skip(h, x_target)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x_target = x[:out_adj.size[1]]
        x = self.out_conv((x, x_target), out_adj.edge_index)
    
        return x

    def inference(self, x, loader, device):
        
        # we do not use dropout because inferense is test
        x = conv_for_gpumemory(x, loader, self.in_conv, device)
        x = F.relu(x)

        for mid_conv, skip in zip(self.mid_convs, self.skips):
            h = conv_for_gpumemory(x, loader, mid_conv, device)
            h = F.relu(h)
            x = skip(h, x)

        x = conv_for_gpumemory(x, loader, self.out_conv, device)
        
        return x


def return_net(cfg):
    # our algorithm (summarize skip-connection)
    if cfg.skip_connection == 'summarize': 
        if cfg.base_gnn == 'GCN':
            raise NotImplementedError()
        elif cfg.base_gnn == 'SAGE':
            return SummarizeSAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            raise NotImplementedError()

    # existing algorithms ([vanilla, res, dense, highway] skip-connection)
    else:
        if cfg.base_gnn == 'GCN':
            raise NotImplementedError()
        elif cfg.base_gnn == 'SAGE':
            return SAGE(cfg)
        elif cfg.base_gnn == 'GAT':
            raise NotImplementedError()
