# Generalizable Symbolic Optimizer Learning, ECCV 2024.

This is the code implementation for the paper "Generalizable Symbolic Optimizer Learning".

<p align="center">
<img src="https://github.com/songxt3/songxt3.github.io/blob/main/images/SOL_framework.png">
</p>

### Requirements

- Python 3.7
- Pytorch 1.13.1
- torch_geometric 2.3.1
- transformer 4.28.1
- robustbench
- torchattacks

## Data
Image classification
- MNIST
- CIFAR-10

GNN node classification
- CiteSeer
- Cora
- PubMed

BERT finetuning
- MRPC
- WNLI
- SST-2
- Cola
- RTE

## Running the Experiments
To perform the experiment on MNIST with MNISTNET
```
cd ./convnet
python -u mnistnet.py --max_epoch 50 --optimizer_steps 100 --truncated_bptt_step 20 --updates_per_epoch 10 --batch_size 128
```
To perform the experiment on CIFAR-10 with ConvNet
```
cd ./convnet
python -u convnet.py --max_epoch 50 --optimizer_steps 100 --truncated_bptt_step 20 --updates_per_epoch 10 --batch_size 64
```
To perform the experiment on adversarial attacks
```
cd ./attack
python -u train.py
```
To perform the experiment on GNN training
```
cd ./gnn
python -u main.py
```
To perform the experiement on BERT finetuning, we use Cola as an example, the other datasets are similar
```
cd ./bert
python -u main_cola.py
```
For SST-2 and RTE datasets, test the learned optimizer using new_sst.py.

The MRPC experiment is executed in separate codes
```
cd ./bert
python -u MRPC_train.py -maxlen 64 --max_epoch 100 --updates_per_epoch 10 --optimizer_steps 150 --truncated_bptt_step 30 > mrpc_log 2>&1 &ls
```

## Reference
```
Wait update.
```