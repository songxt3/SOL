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
from meta_optimizer import ExprOptimizer, SymbolicOptimizer, MetaModel
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
        meta_optimizer.to('cuda:1')
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
    parser.add_argument('-max_eps', type=int, default=5)
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

    # WNLI Dataset
    train_set = WNLIDataset(filename='./data/WNLI/train.tsv', maxlen=args.maxlen)
    val_set = WNLIDataset(filename='./data/WNLI/dev.tsv', maxlen=args.maxlen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=5)
    print("Done in {} seconds".format(time.time() - st))

    expr = learn(criterion, train_loader, args)
    # expr = '-0.0376123*1 - 0.0847966*2 + 0.165415*ad - 0.0296384*ag + 0.0538851*cd - 0.0154843*g - 0.235282*g2 - 0.128663*g3 + 0.0662841*ld + 0.102474*m + 0.0801733*rs + 0.172163*sg + 0.213331*sm - 0.117112*v + 0.0523589*(-0.900845*1 - 0.443723*2 - ad + 0.887758*ag - 0.342368*cd - 0.702426*g + 0.869759*g2 + 0.485333*g3 - 0.492173*ld - 0.358953*m - 0.158045*rs - 0.195202*sg - 0.81565*sm + 0.633661*v)**2 - 1.05435109138489*(-0.222657*1 - 0.258739*2 + 0.0369745*ad + 0.208619*ag - 0.0433732*cd + 0.0700114*g + 0.163197*g2 + 0.0511004*g3 - 0.286737*ld + 0.0158489*m + 0.0413001*rs - 0.0455919*sg + 0.0183253*sm - 0.179492*v)*(0.0439876*1 - 0.0785299*2 + 0.264713*ad - 0.0838345*ag + 0.0233142*cd + 0.141458*g - 0.254214*g2 - 0.0811921*g3 - 0.0274652*ld + 0.192847*m + 0.25285*rs + 0.286161*sg + 0.25375*sm - 0.103268*v) - 0.705232364364171*exp(-0.00653539*1 + 0.0401041*2 - 0.116104*ad + 0.0913967*ag + 0.0877471*cd - 0.0310386*g + 0.0212659*g2 + 0.0504415*g3 + 0.0653937*ld - 0.0381931*m - 0.222908*rs - 0.254865*sg - 0.105901*sm + 0.199825*v) + 1.39194*sign(0.00290239*m - 0.00158121*sm) + 0.113942963309789 - 1.10408/(exp(1.5274*1 + 0.662694*2 + 4.20403*ad - 0.9212*ag - 1.6852*cd + 2.15362*g - 2.03231*g2 - 0.705994*g3 - 0.517564*ld + 0.66836*m + 3.76041*rs + 4.34375*sg + 4.00287*sm - 1.2533*v) + 1)'

    print("Let the training begin")
    st = time.time()
    print('--------------Expr--------------------------------')
    expr_train(net_2, criterion, expr, train_loader, val_loader, args)
    print('--------------AdamW--------------------------------')
    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))
