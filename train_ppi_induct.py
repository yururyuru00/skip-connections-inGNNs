import hydra
from omegaconf import DictConfig
from sklearn.metrics import f1_score
from tqdm import tqdm
from hydra import utils

import torch
from torch_scatter import scatter
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from models import return_net


def train(loader, model, optimizer, device):
    # train
    model.train()
    criteria = torch.nn.BCEWithLogitsLoss()

    for batch_id, data in enumerate(loader): # in [g1, g2, ..., g20]
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criteria(out, data.y)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(loader, model, device):
    model.eval()

    ys, preds = [], []
    for data in loader: # only one graph (=g1+g2)
        data = data.to(device)
        ys.append(data.y)
        out = model(data.x, data.edge_index)
        preds.append((out > 0).float().cpu())

    y    = torch.cat(ys, dim=0).to('cpu').detach().numpy().copy()
    pred = torch.cat(preds, dim=0).to('cpu').detach().numpy().copy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


def run(cfg, data_loader, device):
    train_loader, val_loader, test_loader = data_loader

    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(train_loader, model, optimizer, device)
    test_acc = test(test_loader, model, device)

    return test_acc


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg = cfg[cfg.key]
    print(cfg)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = utils.get_original_cwd() + '/data/' + cfg.dataset
    train_dataset = PPI(root, split='train')
    val_dataset   = PPI(root, split='val')
    test_dataset  = PPI(root, split='test')

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    data_loader = [train_loader, val_loader, test_loader]

    test_acces = []
    for tri in range(cfg['n_tri']):
        test_acc = run(cfg, data_loader, device)
        test_acces.append(test_acc)
    print('mean test acc ({} tri): {:.4f}'.format(cfg.n_tri, sum(test_acces)/len(test_acces)))
    
    
if __name__ == "__main__":
    main()