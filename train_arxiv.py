import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from hydra import utils

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from models import return_net


def train(data, model, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    out = out.log_softmax(dim=-1)
    out = out[data['train_mask']]
    loss = F.nll_loss(out, data.y.squeeze(1)[data['train_mask']])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(data, model, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)
    
    mask = data['test_mask']
    test_acc = evaluator.eval({
        'y_true': data.y[mask],
        'y_pred': y_pred[mask],
    })['acc']
    
    return test_acc


def run(cfg, data, device):
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])
    evaluator = Evaluator('ogbn-arxiv')

    for epoch in tqdm(range(1, cfg['epochs'])):
        train(data, model, optimizer)
    test_acc = test(data, model, evaluator)

    return test_acc


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg = cfg[cfg.key]
    print(cfg)
    
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = utils.get_original_cwd() + '/data/' + cfg.dataset
    dataset = PygNodePropPredDataset('ogbn-arxiv', root, transform=T.ToSparseTensor())
    splitted_idx = dataset.get_idx_split()
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    # Set split indices to masks.
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[splitted_idx[split]] = True
        data[f'{split}_mask'] = mask

    test_acces = []
    for tri in range(cfg['n_tri']):
        test_acc = run(cfg, data, device)
        test_acces.append(test_acc)
    print('mean test acc ({} tri): {:.4f}'.format(cfg.n_tri, sum(test_acces)/len(test_acces)))
    

if __name__ == "__main__":
    main()