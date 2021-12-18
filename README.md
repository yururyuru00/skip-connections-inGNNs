# skipconnections-inGNNs
python scripts for my academic research#2

## Requirements
```bash
torch
torch_geometric
hydra
tqdm
```

## Models 
| Skip | Vanilla | Res      | Dense      | Highway      | Complex Summarize | Simple Summarize[ours] |
|------|---------|----------|----------  |--------------|-------------------|------------------|
| GCN  | GCN     | Res-GCN  | Dense-GCN  | Highway-GCN  | CS-GCN            | SS-GCN           |
| GAT  | GAT     | Res-GAT  | Dense-GAT  | Highway-GAT  | CS-GAT            | SS-GAT           |
| SAGE | SAGE    | Res-SAGE | Dense-SAGE | Highway-SAGE | CS-SAGE           | SS-SAGE          |

## Guide to experimental replication
PubMed dataset with Simple Summarize-GNN (n_layer=8)
```bash 
python3 train_planetoid.py key=SummarizeGCN_PubMed SummarizeGCN_PubMed.n_layer=8
```

PPI dataset with Simple Summarize-GNN (n_layer=6)
```bash 
python3 train_ppi.py key=SummarizeSAGE_PPI SummarizeSAGE_PPI.n_layer=6
```

PPI_induct. dataset with Simple Summarize-GNN (n_layer=6)
```bash 
python3 train_ppi_induct.py key=SummarizeGAT_PPIinduct SummarizeGAT_PPIinduct.n_layer=6
```

Reddit dataset with Simple Summarize-GNN (n_layer=6)
```bash 
python3 train_reddit.py key=SummarizeSAGE_Reddit_tuned SummarizeSAGE_Reddit_tuned.n_layer=6
```

Arxiv dataset with Simple Summarize-GNN (n_layer=6)
```bash 
python3 train_arxiv.py key=SummarizeGCN_Arxiv SummarizeGCN_Arxiv.n_layer=6
```

If you need to know the parameters in detail, please check conf/config.yaml.


