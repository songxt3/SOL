from torch.utils.data import DataLoader
from MRPC_dataloader import MRPCDataset
import argparse
import torch
from MRPC_model import MRPCModel, BERT, Classifier
import copy
import torch.nn as nn
import torch.optim as optim
from transformers import BertConfig
import math
import numpy as np
from sklearn.metrics import f1_score
from download_glue_data import format_mrpc
import functions
from meta_optimizer import SymbolicOptimizer, MetaModel, ExprOptimizer
from itertools import cycle
from torch.autograd import Variable
import pretty_print
from tqdm import tqdm

torch.manual_seed(63)
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-maxlen', type=int, default=256)
parser.add_argument('-batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--updates_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

format_mrpc(data_dir='./data/', path_to_data='./data/MRPC/')

train_set = MRPCDataset(filename='./data/MRPC/train.tsv', maxlen=args.maxlen)
val_set = MRPCDataset(filename='./data/MRPC/dev.tsv', maxlen=args.maxlen)

train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=5)


def compute_acc(model):
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for data in val_loader:
            seq, attn_masks, labels = data
            if args.cuda:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            outputs = model(seq, attn_masks)
            predictions = (outputs.squeeze() > 0.5).long()
            true_labels.extend(labels.cpu().numpy().flatten())
            predicted_labels.extend(predictions.cpu().numpy().flatten())
    # Convert the lists to NumPy arrays for compatibility with sklearn
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    # Calculate F1 score
    f1 = f1_score(true_labels, predicted_labels, average='binary')
    print('Accuracy on the test set: {:.2f}'.format(f1))
    return f1


def train(model, optimizer, criterion, num_epochs, lr):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batch = 0
        for i, data in enumerate(train_loader):
            seq, attn_masks, labels = data
            if args.cuda:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

            outputs = model(seq, attn_masks)
            loss = criterion(outputs.squeeze(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
            num_batch += 1

        # print(f'Epoch {epoch + 1}: Loss {running_loss / num_batch}%')

    accuracy = compute_acc(model)
    print('Accuracy:', accuracy)
    return accuracy


def cosine_annealing_lr(epoch, T_max, eta_min, eta_max):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (epoch % T_max) / T_max))


def expr_train(model, expr_optimizer, criterion, num_epochs, lr):
    model.train()
    for epoch in range(num_epochs):
        lr = cosine_annealing_lr(epoch, num_epochs, eta_min=lr / 10, eta_max=lr)
        running_loss = 0.0
        num_batch = 0
        for i, data in enumerate(train_loader):
            seq, attn_masks, labels = data
            if args.cuda:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()

            outputs = model(seq, attn_masks)
            loss = criterion(outputs.squeeze(), labels.float())
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            expr_optimizer.meta_update(model, epoch + 1, num_epochs, lr=lr)
            running_loss += loss
            num_batch += 1

        # print(f'Epoch {epoch + 1}: Loss {running_loss / num_batch}%')

    accuracy = compute_acc(model)
    print('Accuracy:', accuracy)
    return accuracy


def main(lr):
    config = BertConfig(
        hidden_size=768,  # 隐藏层维度
    )
    # meta_model = SST2Model(config)
    meta_model = Classifier()
    if args.cuda:
        meta_model.cuda()

    funcs = [
        *[functions.Constant()],
        *[functions.Identity()],
        *[functions.Square()],
        *[functions.Exp()],
        *[functions.Sigmoid()],
        *[functions.Sign()],
        *[functions.Product(1.0)],
    ]
    init_sd_first = 0.1
    init_sd_last = 1.0
    init_sd_middle = 0.5
    x_dim = 14
    width = len(funcs)
    n_double = functions.count_double(funcs)
    meta_optimizer = SymbolicOptimizer(MetaModel(meta_model), n_layers=1, in_dim=14, funcs=funcs, initial_weights=[
        # kind of a hack for truncated normal
        torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
        torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
        torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
        torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
    ])
    if args.cuda:
        meta_optimizer.cuda()

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)
    criterion = nn.BCELoss()
    train_iter = cycle(train_loader)

    bert_model = BERT(config)
    if args.cuda:
        bert_model.cuda()

    for epoch in tqdm(range(args.max_epoch), desc='Training:'):
        decrease_in_loss = 0.0
        final_loss = 0.0
        for i in range(args.updates_per_epoch):
            # model = SST2Model(config)
            model = Classifier()
            if args.cuda:
                model.cuda()

            seq, attn_masks, labels = next(train_iter)
            if args.cuda:
                seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
            seq, attn_masks, labels = Variable(seq), Variable(attn_masks), Variable(labels)

            # outputs = model(seq, attn_masks)
            cls_output = bert_model(seq, attn_masks)
            cls_output = Variable(cls_output)
            outputs = model(cls_output)
            initial_loss = criterion(outputs.squeeze(), labels.float())

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                meta_optimizer.reset(keep_states=k > 0, model=model, params=model.parameters())

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda()
                for j in range(args.truncated_bptt_step):
                    seq, attn_masks, labels = next(train_iter)
                    if args.cuda:
                        seq, attn_masks, labels = seq.cuda(), attn_masks.cuda(), labels.cuda()
                    seq, attn_masks, labels = Variable(seq), Variable(attn_masks), Variable(labels)

                    # outputs = model(seq, attn_masks)
                    cls_output = bert_model(seq, attn_masks)
                    cls_output = Variable(cls_output)
                    outputs = model(cls_output)
                    loss = criterion(outputs.squeeze(), labels.float())
                    model.zero_grad()
                    loss.backward()
                    # Gradient cropping to avoid gradient explosions
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, step=(k * args.truncated_bptt_step + j + 1),
                                                            T=args.optimizer_steps, lr=lr)

                    # outputs = meta_model(seq, attn_masks)
                    outputs = meta_model(cls_output)
                    loss = criterion(outputs.squeeze(), labels.float())

                    loss_sum += (loss - Variable(prev_loss))
                    prev_loss = loss.data

                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                    if torch.isnan(param.grad).any():
                        print(torch.isnan(param.grad))
                        param.grad.data = torch.where(torch.isnan(param.grad.data), torch.full_like(param.grad.data, 0),
                                                      param.grad.data)
                optimizer.step()

            # Compute relative decrease in the loss function
            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()

        # print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch,
        #                                                                               final_loss / args.updates_per_epoch,
        #                                                                               decrease_in_loss / args.updates_per_epoch))
    with torch.no_grad():
        weights = meta_optimizer.get_weights()
        expr = pretty_print.network(weights, funcs,
                                    ["m", "v", "g", "g2", "g3", "ag", "sg", "sm", "ad", "rs", "ld", "cd", "1",
                                     "2"])

    return expr


def eval_optimizer(expr, lr):
    config = BertConfig(
        hidden_size=768,  # 隐藏层维度
    )
    model_sol = MRPCModel(config)
    if args.cuda:
        model_sol.cuda()
    model_adam = copy.deepcopy(model_sol)
    if args.cuda:
        model_adam.cuda()
    criterion = nn.BCELoss()

    optimizer = optim.AdamW(model_adam.parameters(), betas=(0.95, 0.95), lr=lr)

    acc_adam = train(model=model_adam, optimizer=optimizer, criterion=criterion, num_epochs=5, lr=lr)

    expr_optimizer = ExprOptimizer(expr)
    expr_optimizer.reset(params=model_sol.dense.parameters())
    acc_sol = expr_train(model=model_sol, expr_optimizer=expr_optimizer, criterion=criterion, num_epochs=5, lr=lr)

    print("Accuracy by SOL:", acc_sol)
    print("Accuracy by Adam:", acc_adam)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    lr = 5e-4
    expr = main(lr=lr)
    # expr = '-0.010037*1 - 0.0165584*2 + 0.0470936*ad - 0.0208686*cd + 0.0632879*g - 0.10127*g2 + 0.0615034*g3 + 0.0230904*ld + 0.0882166*m + 0.0845613*rs + 0.0490683*sg + 0.0810729*sm - 0.0842005*v + 0.0643081*(-0.840519*1 - 0.46178*2 - ad + 0.840838*ag - 0.564785*cd - 0.73048*g - 0.738461*g2 - 0.150937*g3 - 0.319391*ld - 0.192764*m - 0.200511*rs - 0.776264*sg - 0.341622*sm - 0.291858*v)**2 + 2.03392148017883*(0.0158682*1 - 0.0866373*2 + 0.236099*ad - 0.0195902*ag - 0.0149681*cd + 0.113531*g - 0.235193*g2 + 0.192015*g3 - 0.0125626*ld + 0.262811*m + 0.172704*rs + 0.212438*sg + 0.248831*sm - 0.0611465*v)*(0.0862622*1 + 0.171756*2 + 0.0288025*ad - 0.163132*ag + 0.232837*cd - 0.0494352*g + 0.181913*g2 - 0.119592*g3 + 0.171832*ld - 0.116764*m - 0.0562006*rs + 0.0872317*sg - 0.0983446*sm - 0.0603059*v) - 0.149987935416945*exp(0.111106*1 + 0.149787*2 - 0.200283*ad - 0.0415903*ag + 0.0650179*cd - 0.220194*g + 0.112761*g2 - 0.148106*g3 + 0.0416328*ld - 0.0523462*m - 0.147251*rs - 0.182519*sg - 0.134201*sm + 0.189336*v) + 0.224269*sign(-0.0164683*ag + 0.0364517*g2 + 0.0284836*g3 + 0.0365508*rs) + 0.327721133225211 - 0.80355/(exp(0.311639*1 - 0.941267*2 + 1.8612*ad + 0.746803*ag - 0.805005*cd - 0.485736*g2 + 0.656839*g3 - 1.67365*ld + 1.23344*m + 1.27884*rs + 3.33465*sg + 1.36153*sm - 2.46814*v) + 1)'
    eval_optimizer(expr, lr=lr)
