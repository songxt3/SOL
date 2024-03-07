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
from dataloader import COLADataset
from meta_optimizer import ExprOptimizer, SimpleSymbolicOptimizer, MetaModel, Solo
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef

torch.manual_seed(111)
torch.multiprocessing.set_sharing_strategy('file_system')
enable_gpu = True


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.5).long()
    # acc = (soft_probs.squeeze() == labels).float().mean()
    mcc = matthews_corrcoef(labels.cpu(), soft_probs.squeeze().cpu())
    return mcc


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
    x_dim = 12
    width = len(funcs)
    n_double = functions.count_double(funcs)
    meta_optimizer = SimpleSymbolicOptimizer(MetaModel(meta_model), n_layers=1, in_dim=12, funcs=funcs, initial_weights=[
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
    for epoch in tqdm(range(50), desc='Training:'):
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

            length = 100
            unrolled_len = 20
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

        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch,
                                                                                      final_loss / 10,
                                                                                      decrease_in_loss / 10))

    # test process
    with torch.no_grad():
        weights = meta_optimizer.get_weights()
        expr = pretty_print.network(weights, funcs,
                                    ["m", "v", "g", "g2", "g3", "ag", "sg", "sm", "ad", "rs", "1",
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

    print("Expr best accuracy:", best_acc)


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
    print("AdamW best accuracy:", best_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type=int, default=64)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-lr', type=float, default=1e-5)
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

    # cola Dataset
    train_set = COLADataset(filename='./data/cola_public/raw/in_domain_train.tsv', maxlen=args.maxlen)
    val_set = COLADataset(filename='./data/cola_public/raw/in_domain_dev.tsv', maxlen=args.maxlen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=5)
    print("Done in {} seconds".format(time.time() - st))

    expr = learn(criterion, train_loader, args)
    # expr = '0.110899*1 + 0.103472*2 + 0.0982446*ad - 0.0884523*ag + 0.0732069*g + 0.0508218*g3 + 0.102418*m + 0.070173*rs + 0.0392048*sg + 0.0500689*sm - 0.0077598*v + 0.658640623092651*(-0.318187*1 - 0.186689*2 - 0.24335*ad + 0.0657449*ag - 0.0844163*g - 0.058631*g2 - 0.0831089*g3 - 0.227758*m - 0.162449*rs - 0.222419*sg - 0.185061*sm - 0.108678*v)*(-0.0861372*1 - 0.316076*2 - 0.184817*ad + 0.223552*ag - 0.154475*g - 0.0905247*g2 - 0.0432197*g3 - 0.159137*m - 0.187158*rs - 0.201425*sg - 0.153528*sm - 0.0879749*v) - 0.00971104*(0.048973*1 - 0.0553442*2 + 0.121515*ad - 0.0549982*ag + 0.712347*g - 0.0823261*g2 + 0.0161293*g3 + 0.457588*m + 0.0785416*rs - 0.0222113*sg - 0.0902402*sm - v)**2 - 0.245218849174842*exp(-0.11537*1 - 0.218823*2 - 0.0238919*ad + 0.129959*ag - 0.0904845*g + 0.0201965*g2 - 0.0104821*g3 - 0.22301*m - 0.0506201*rs - 0.115658*sg - 0.030156*sm + 0.0175739*v) + 0.252095*sign(g) + 1.43253049849735 + 0.634623/(exp(-2.62307*1 - 3.62448*2 - 1.71192*ad - 0.504678*ag - 1.99911*g + 0.0960216*g2 - 0.574406*g3 - 1.90638*m - 3.60831*rs - 0.187205*sg - 0.269201*sm - 0.151803*v) + 1)'
    solo_optimizer = Solo(net_2.parameters(), lr=args.lr, expr=expr)

    print("Let the training begin")
    st = time.time()
    print('--------------Expr--------------------------------')
    # expr_train(net_2, criterion, expr, train_loader, val_loader, args)
    train(net_2, criterion, solo_optimizer, train_loader, val_loader, args)
    print('--------------AdamW--------------------------------')
    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))
