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
from meta_optimizer import ExprOptimizer, SimpleSymbolicOptimizer, MetaModel, Solo
from torch.autograd import Variable
from tqdm import tqdm

torch.manual_seed(1388)

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
        meta_optimizer.to('cuda:1')
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
    print("Expr best accuracy:", best_acc.item() * 100)


def train(net, criterion, optimizer, scheduler, train_loader, val_loader, args):
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
            # scheduler.step()

            if it % args.print_every == 0:
                acc = get_accuracy_from_logits(logits, labels)
                print("Iteration {} of epoch {} complete. Loss : {} Accuracy : {}".format(it, ep, loss.item(), acc))

        val_acc, val_loss = evaluate(net, criterion, val_loader, args)
        net.train()
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
    print("Best accuracy:", best_acc.item() * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=1)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type=int, default=25)
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opti, T_max=100, eta_min=5e-5)
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

    expr = learn(criterion, train_loader, args)
    # expr = '-0.010037*1 - 0.0165584*2 + 0.0470936*ad - 0.0208686*cd + 0.0632879*g - 0.10127*g2 + 0.0615034*g3 + 0.0230904*ld + 0.0882166*m + 0.0845613*rs + 0.0490683*sg + 0.0810729*sm - 0.0842005*v + 0.0643081*(-0.840519*1 - 0.46178*2 - ad + 0.840838*ag - 0.564785*cd - 0.73048*g - 0.738461*g2 - 0.150937*g3 - 0.319391*ld - 0.192764*m - 0.200511*rs - 0.776264*sg - 0.341622*sm - 0.291858*v)**2 + 2.03392148017883*(0.0158682*1 - 0.0866373*2 + 0.236099*ad - 0.0195902*ag - 0.0149681*cd + 0.113531*g - 0.235193*g2 + 0.192015*g3 - 0.0125626*ld + 0.262811*m + 0.172704*rs + 0.212438*sg + 0.248831*sm - 0.0611465*v)*(0.0862622*1 + 0.171756*2 + 0.0288025*ad - 0.163132*ag + 0.232837*cd - 0.0494352*g + 0.181913*g2 - 0.119592*g3 + 0.171832*ld - 0.116764*m - 0.0562006*rs + 0.0872317*sg - 0.0983446*sm - 0.0603059*v) - 0.149987935416945*exp(0.111106*1 + 0.149787*2 - 0.200283*ad - 0.0415903*ag + 0.0650179*cd - 0.220194*g + 0.112761*g2 - 0.148106*g3 + 0.0416328*ld - 0.0523462*m - 0.147251*rs - 0.182519*sg - 0.134201*sm + 0.189336*v) + 0.224269*sign(-0.0164683*ag + 0.0364517*g2 + 0.0284836*g3 + 0.0365508*rs) + 0.327721133225211 - 0.80355/(exp(0.311639*1 - 0.941267*2 + 1.8612*ad + 0.746803*ag - 0.805005*cd - 0.485736*g2 + 0.656839*g3 - 1.67365*ld + 1.23344*m + 1.27884*rs + 3.33465*sg + 1.36153*sm - 2.46814*v) + 1)'
    # solo_optimizer = Solo(net_2.parameters(), lr=args.lr, expr=expr)
    # solo_scheduler = optim.lr_scheduler.CosineAnnealingLR(solo_optimizer, T_max=100, eta_min=5e-5)

    print("Let the training begin")
    st = time.time()

    print('--------------Expr--------------------------------')
    expr_train(net_2, criterion, expr, train_loader, val_loader, args)
    # train(net_2, criterion, solo_optimizer, solo_scheduler, train_loader, val_loader, args)

    print('--------------AdamW--------------------------------')
    train(net, criterion, opti, scheduler, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))
