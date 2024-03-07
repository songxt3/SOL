import copy
from itertools import cycle
import functions
import pretty_print
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time
from model import SentimentClassifier, Classifier, BERT
from dataloader import WNLIDataset, SSTDataset
from meta_optimizer import ExprOptimizer, SymbolicOptimizer, MetaModel, Solo
from torch.autograd import Variable
from tqdm import tqdm

torch.manual_seed(280)
torch.multiprocessing.set_sharing_strategy('file_system')
enable_gpu = True


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc


def evaluate(net, criterion, dataloader, args):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks, labels in dataloader:
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            logits = net(seq, attn_masks)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            count += 1

    return mean_acc / count, mean_loss / count


def learn(criterion, train_loader, args):
    train_iter = cycle(train_loader)

    # meta_model = SentimentClassifier(args.freeze_bert, bert_hidden_size=768, num_classes=1)
    meta_model = Classifier(bert_hidden_size=768, num_classes=1)
    if enable_gpu:
        meta_model.cuda(args.gpu)

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
    if enable_gpu:
        meta_optimizer.to('cuda:0')
    bert_model = BERT()
    bert_model.cuda(args.gpu)
    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)
    for epoch in tqdm(range(100), desc='Training:'):
        decrease_in_loss = 0.0
        final_loss = 0.0
        for i in range(10):
            # model = SentimentClassifier(args.freeze_bert, bert_hidden_size=768, num_classes=1)
            model = Classifier(bert_hidden_size=768, num_classes=1)
            if enable_gpu:
                model.cuda(args.gpu)

            seq, attn_masks, labels = next(train_iter)
            if enable_gpu:
                seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
            cls_output = bert_model(seq, attn_masks)
            cls_output = Variable(cls_output)
            logits = model(cls_output)
            # logits = model(seq, attn_masks)
            initial_loss = criterion(logits.squeeze(-1), labels.float())

            length = 150
            unrolled_len = 30
            for k in range(int(length // unrolled_len)):
                meta_optimizer.reset(keep_states=k > 0, model=model, params=model.linear.parameters())

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if enable_gpu:
                    prev_loss = prev_loss.cuda(args.gpu)
                for j in range(unrolled_len):
                    seq, attn_masks, labels = next(train_iter)
                    if enable_gpu:
                        seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)
                    cls_output = bert_model(seq, attn_masks)
                    cls_output = Variable(cls_output)
                    f_x = model(cls_output)
                    # f_x = model(seq, attn_masks)
                    loss = criterion(f_x.squeeze(-1), labels.float())
                    model.zero_grad()
                    loss.backward()
                    # 梯度裁剪，避免出现梯度爆炸情况
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    meta_model = meta_optimizer.meta_update(model, step=k * unrolled_len + j + 1,
                                                            T=length * unrolled_len, lr=args.lr)

                    f_x = meta_model(cls_output)
                    # f_x = meta_model(seq, attn_masks)
                    loss = criterion(f_x.squeeze(-1), labels.float())

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

            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()

    # test process
    with torch.no_grad():
        weights = meta_optimizer.get_weights()
        expr = pretty_print.network(weights, funcs,
                                    ["m", "v", "g", "g2", "g3", "ag", "sg", "sm", "ad", "rs", "ld", "cd", "1",
                                     "2"], threshold=0.001)

    return expr


def expr_train(net, criterion, expr, train_loader, val_loader, args):
    best_acc = 0
    expr_optimizer = ExprOptimizer(expr)
    expr_optimizer.reset(params=net.linear.parameters())
    for ep in range(args.max_eps):

        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            # Clear gradients
            net.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            expr_optimizer.meta_update(net, ep + 1, args.max_eps, lr=args.lr)

            if it % args.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it, ep, loss.item(), acc))

        val_acc, val_loss = evaluate(net, criterion, val_loader, args)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
            # torch.save(net.state_dict(), './Models/sst_expr_{}_freeze_{}.dat'.format(ep, args.freeze_bert))
    print("Expr best accuracy:", best_acc.item() * 100)

def train(net, criterion, optimizer, train_loader, val_loader, args):
    best_acc = 0
    for ep in range(args.max_eps):

        for it, (seq, attn_masks, labels) in enumerate(train_loader):
            # Clear gradients
            net.zero_grad()
            # Converting these to cuda tensors
            seq, attn_masks, labels = seq.cuda(args.gpu), attn_masks.cuda(args.gpu), labels.cuda(args.gpu)

            # Obtaining the logits from the model
            logits = net(seq, attn_masks)

            # Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            # Backpropagating the gradients
            loss.backward()

            # Optimization step
            optimizer.step()

            if it % args.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it, ep, loss.item(), acc))

        val_acc, val_loss = evaluate(net, criterion, val_loader, args)
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
    print("AdamW best accuracy:", best_acc.item() * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=1)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type=int, default=25)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-lr', type=float, default=2e-5)
    parser.add_argument('-print_every', type=int, default=100)
    parser.add_argument('-max_eps', type=int, default=3)
    args = parser.parse_args()

    # Instantiating the classifier model
    print("Building model! (This might take time if you are running this for first time)")
    st = time.time()
    net = SentimentClassifier(args.freeze_bert, bert_hidden_size=768, num_classes=1)
    if enable_gpu:
        net.cuda(args.gpu)  # Enable gpu support for the model
    net_2 = copy.deepcopy(net)
    print("Done in {} seconds".format(time.time() - st))

    print("Creating criterion and optimizer objects")
    st = time.time()
    criterion = nn.BCEWithLogitsLoss()
    opti = optim.AdamW(net.parameters(), lr=args.lr)
    print("Done in {} seconds".format(time.time() - st))

    # Creating dataloaders
    print("Creating train and val dataloaders")
    st = time.time()
    '''
    # WNLI Dataset
    train_set = WNLIDataset(filename='./data/WNLI/train.tsv', maxlen=args.maxlen)
    val_set = WNLIDataset(filename='./data/WNLI/dev.tsv', maxlen=args.maxlen)
    '''
    # SST Dataset
    train_set = SSTDataset(filename='./data/SST-2/train.tsv', maxlen=args.maxlen)
    val_set = SSTDataset(filename='./data/SST-2/dev.tsv', maxlen=args.maxlen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=5)
    print("Done in {} seconds".format(time.time() - st))

    # expr = learn(criterion, train_loader, args)
    # expr = '0.0856641*1 + 0.0140159*2 - 0.0235698*ad - 0.104592*ag + 0.126996*cd + 0.137472*g + 0.00815836*g2 + 0.0555743*g3 + 0.0166942*ld + 0.0609867*m + 0.122648*rs + 0.120776*sg + 0.147618*sm + 0.0703643*v + 1.5868935585022*(0.182658*1 + 0.234739*2 + 0.269682*ad - 0.136937*ag + 0.106465*cd + 0.140713*g + 0.0871234*g3 + 0.069266*ld + 0.234922*rs + 0.174586*sg + 0.227891*sm + 0.172915*v)*(0.221285*1 + 0.148046*2 + 0.0645579*ad - 0.163969*ag + 0.166702*cd + 0.0335867*g + 0.0670569*g2 + 0.0908661*g3 + 0.103034*ld + 0.0171128*m + 0.257652*rs + 0.153765*sg + 0.164328*sm + 0.122796*v) + 0.0452597*(-0.781574*1 - 0.530418*2 - ad + 0.572112*ag - 0.972454*cd - 0.542745*g + 0.0449121*g2 - 0.407078*g3 - 0.775676*ld - 0.245183*m - 0.732933*rs - 0.827963*sg - 0.760396*sm - 0.523278*v)**2 - 0.575188425955877*exp(-0.0895223*2 - 0.0913822*ad + 0.043849*ag - 0.122979*cd - 0.134513*g - 0.0306758*g2 - 0.03982*g3 - 0.0318734*ld - 0.123538*m - 0.0458857*rs - 0.149061*sg - 0.118522*sm + 0.0415355*v) + 0.951078*sign(1) - 0.690037938179866 - 0.406531/(exp(0.903318*1 + 2.72211*2 + 3.27853*ad - 2.8636*ag + 0.22331*g + 0.357838*g2 + 2.11985*m + 3.1604*rs + 0.405406*sg + 2.50013*sm) + 1)'
    expr = '0.700751*g + 0.576713*m + 0.515641*v - 0.0061018*(-0.75904*g - 0.658663*m + v)**2 - 0.502744408614215*exp(-0.200952*g - 0.233579*m - 0.159815*v) + 0.514458*sign(m) + 0.201168151862201 + 0.557738/(exp(-5.2956*g - 4.39645*m - 2.50896*v) + 1)'
    solo_optimizer = Solo(net_2.parameters(), lr=args.lr, expr=expr)

    print("Let the training begin")
    st = time.time()
    print('--------------Expr--------------------------------')
    # expr_train(net_2, criterion, expr, train_loader, val_loader, args)
    train(net_2, criterion, solo_optimizer, train_loader, val_loader, args)
    print('--------------AdamW--------------------------------')
    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))
