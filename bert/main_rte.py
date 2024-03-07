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
from dataloader import RTEDataset
from meta_optimizer import ExprOptimizer, SimpleSymbolicOptimizer, MetaModel, Solo
from torch.autograd import Variable
from tqdm import tqdm

torch.manual_seed(25)
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
        net.train()
        print("Epoch {} complete! Validation Accuracy : {}, Validation Loss : {}".format(ep, val_acc, val_loss))
        if val_acc > best_acc:
            print("Best validation accuracy improved from {} to {}, saving model...".format(best_acc, val_acc))
            best_acc = val_acc
    print("AdamW best accuracy:", best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=int, default=1)
    parser.add_argument('-freeze_bert', action='store_true')
    parser.add_argument('-maxlen', type=int, default=128)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-lr', type=float, default=5e-5)
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

    # rte Dataset
    train_set = RTEDataset(filename='./data/RTE/train.tsv', maxlen=args.maxlen)
    val_set = RTEDataset(filename='./data/RTE/dev.tsv', maxlen=args.maxlen)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=5)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=5)
    print("Done in {} seconds".format(time.time() - st))

    expr = learn(criterion, train_loader, args)
    # expr = '0.0117786*1 - 0.000582272*2 + 0.0642267*ad - 0.0278209*ag + 0.0609039*g + 0.00792243*g2 + 0.00654308*g3 + 0.00447625*m + 0.0332765*rs + 0.0343053*sg + 0.0977111*sm + 0.00549125*v + 0.000788384*(-0.727733*1 + 0.320973*2 + 0.507005*ad + ag - 0.0753359*g - 0.263634*m + 0.125063*rs - 0.77929*sg - 0.587381*sm + 0.0449483*v)**2 + 0.847640991210938*(0.00664244*1 - 0.00723596*2 + 0.135008*ad + 0.0247901*ag + 0.0362407*g - 0.00450618*g2 + 0.00825409*g3 + 0.067325*m + 0.0941577*rs + 0.102553*sg + 0.158247*sm + 0.0124416*v)*(0.195097*1 + 0.088762*2 - 0.063345*ad - 0.223571*ag - 0.0145187*g - 0.00646708*g2 + 0.00448548*g3 + 0.0140364*m + 0.029944*rs - 0.00861674*sg - 0.0135936*sm + 0.130794*v) - 0.402384847401704*exp(-0.0156423*1 + 0.0129938*2 - 0.0216611*ad - 0.059775*ag + 0.0115238*g - 0.0469424*g2 - 0.116046*m - 0.124361*rs - 0.0407348*sg - 0.132727*sm - 0.0596081*v) - 0.202378*sign(0.0015269*1 - 0.0378086*g - 0.013156*m - 0.00900533*sm) + 1.72200176119718 - 1.18457/(exp(-1.31462*2 + 1.77828*ad + 0.241781*ag + 1.26038*g + 0.740951*g2 + 0.160378*g3 + 1.83776*m + 1.602*rs + 2.13443*sg + 0.905846*sm + 0.681621*v) + 1)'
    solo_optimizer = Solo(net_2.parameters(), lr=args.lr, expr=expr)


    print("Let the training begin")
    st = time.time()
    print('--------------Expr--------------------------------')
    # expr_train(net_2, criterion, expr, train_loader, val_loader, args)
    train(net_2, criterion, solo_optimizer, train_loader, val_loader, args)
    print('--------------AdamW--------------------------------')
    train(net, criterion, opti, train_loader, val_loader, args)
    print("Done in {} seconds".format(time.time() - st))
