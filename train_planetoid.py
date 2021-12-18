import hydra
from hydra import utils
from tqdm import tqdm
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

from models import return_net
from utils import accuracy


def train(data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    # train by class label
    h = model(data.x, data.edge_index)
    prob_labels = F.log_softmax(h, dim=1)
    loss_train  = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    h = model(data.x, data.edge_index)
    prob_labels_val = F.log_softmax(h, dim=1)
    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])

    return loss_val.item()


def test(data, model):
    model.eval()
    h = model(data.x, data.edge_index)
    prob_labels_test = F.log_softmax(h, dim=1)
    acc = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])

    return acc


def run(cfg, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = return_net(cfg).to(device)
    optimizer = torch.optim.Adam(params       = model.parameters(), 
                                 lr           = cfg['learning_rate'], 
                                 weight_decay = cfg['weight_decay'])

    best_loss = 100.
    bad_counter = 0
    for epoch in range(1, cfg['epochs']):
        loss_val = train(data, model, optimizer)

        if loss_val < best_loss:
            best_loss = loss_val
            bad_counter = 0
        else:
            bad_counter += 1
        if bad_counter == cfg['patience']:
            break

    test_acc = test(data, model)
    return test_acc


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    cfg = cfg[cfg.key]
    print(cfg)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = utils.get_original_cwd() + '/data/' + cfg.dataset
    dataset = Planetoid(root          = root,
                        name          = cfg['dataset'],
                        split         = cfg['split'],
                        transform     = eval(cfg['transform']),
                        pre_transform = eval(cfg['pre_transform']))
    data = dataset[0].to(device)

    test_acces = []
    for tri in tqdm(range(cfg['n_tri'])):
        test_acc = run(cfg, data)
        test_acces.append(test_acc)
    print('mean test acc ({} tri): {:.4f}'.format(cfg.n_tri, sum(test_acces)/len(test_acces)))
    

if __name__ == "__main__":
    main()