import hydra
from mlflow.models import model
from omegaconf import DictConfig
from tqdm import tqdm
from hydra import utils

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.loader import NeighborSampler

from models_for_reddit import return_net


def train(data, train_loader, model, optimizer, device):
    model.train()

    for batch_id, (batch_size, n_id, adjs) in enumerate(train_loader):
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        h = model(data.x[n_id], adjs)
        prob_labels = F.log_softmax(h, dim=1)
        loss = F.nll_loss(prob_labels, data.y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(data, test_loader, model, device):
    model.eval()

    h = model.inference(data.x, test_loader, device)
    y_true = data.y.unsqueeze(-1)
    y_pred = h.argmax(dim=-1, keepdim=True)
    test_acc = int(y_pred[data.test_mask].eq(y_true[data.test_mask]).sum()) / int(data.test_mask.sum())
    
    return test_acc


def run(cfg, data, train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    
    for epoch in tqdm(range(1, cfg['epochs'])):
        train(data, train_loader, model, optimizer, device)
    test_acc = test(data, test_loader, model, device)

    return test_acc


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg = cfg[cfg.key]
    print(cfg)
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = utils.get_original_cwd() + '/data/' + cfg.dataset
    dataset = Reddit(root)
    data = dataset[0].to(device)

    sizes_l = [25, 10, 10, 10, 10, 10] # partial aggregate when train
    sizes_test = [-1]                  # full aggregate when test
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=sizes_l[:cfg['n_layer']], batch_size=1024, shuffle=True,
                                   num_workers=0)
    test_loader = NeighborSampler(data.edge_index, node_idx=None,
                                  sizes=sizes_test, batch_size=1024, shuffle=False,
                                  num_workers=0)

    test_acces = []
    for tri in range(cfg['n_tri']):
        test_acc = run(cfg, data, train_loader, test_loader)
        test_acces.append(test_acc)
    print('mean test acc ({} tri): {:.4f}'.format(cfg.n_tri, sum(test_acces)/len(test_acces)))
    

if __name__ == "__main__":
    main()
