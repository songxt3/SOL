import argparse
import copy
import math
import torch
from torchvision import datasets, transforms
from model import ConvNet
import functions
from meta_optimizer import MetaModel, SymbolicOptimizer, ExprOptimizer
import torch.optim as optim
import torch.nn as nn
from itertools import cycle
from torch.autograd import Variable
import pretty_print
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch SOL example')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--updates_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=10000, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.optimizer_steps % args.truncated_bptt_step == 0

torch.manual_seed(66)

# CIFAR-10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.CIFAR10(root='../../AdamTree/datasets', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

test_set = datasets.CIFAR10(root='../../AdamTree/datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
)


def compute_acc(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('Accuracy on the test set: {:.2f}%'.format(accuracy))
    return accuracy


def cosine_annealing_lr(epoch, T_max, eta_min, eta_max):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (epoch % T_max) / T_max))


def expr_train(model, expr_optimizer, criterion, num_epochs, lr):
    model.train()
    print('SOL training begin:')
    for epoch in tqdm(range(num_epochs), desc="Training", unit='epoch'):
        lr = cosine_annealing_lr(epoch, num_epochs, eta_min=lr / 10, eta_max=lr)
        running_loss = 0.0
        num_batch = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            expr_optimizer.meta_update(model, epoch + 1, num_epochs, lr=lr)
            running_loss += loss
            num_batch += 1

        print(f'Epoch {epoch + 1}: Loss {running_loss / num_batch}%')

    accuracy = compute_acc(model)
    return accuracy


def train(model, optimizer, criterion, num_epochs, lr):
    model.train()
    print('Adam training begin:')
    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        running_loss = 0.0
        num_batch = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if args.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
            num_batch += 1

        print(f'Epoch {epoch + 1}: Loss {running_loss / num_batch}%')

    accuracy = compute_acc(model)
    return accuracy


def main(lr):
    meta_model = ConvNet()
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
    criterion = nn.CrossEntropyLoss()
    train_iter = cycle(train_loader)

    for epoch in range(args.max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        for i in range(args.updates_per_epoch):
            # Sample a new model
            model = ConvNet()
            if args.cuda:
                model.cuda()

            x, y = next(train_iter)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # Compute initial loss of the model
            f_x = model(x)
            initial_loss = criterion(f_x, y)

            for k in range(args.optimizer_steps // args.truncated_bptt_step):
                meta_optimizer.reset(keep_states=k > 0, model=model, params=model.parameters())

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if args.cuda:
                    prev_loss = prev_loss.cuda()
                for j in range(args.truncated_bptt_step):
                    x, y = next(train_iter)
                    if args.cuda:
                        x, y = x.cuda(), y.cuda()
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    loss = criterion(f_x, y)
                    model.zero_grad()
                    loss.backward()
                    # Gradient cropping to avoid gradient explosions
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(model, step=(k * args.truncated_bptt_step + j + 1),
                                                            T=args.optimizer_steps, lr=lr)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    loss = criterion(f_x, y)

                    loss_sum += (loss - Variable(prev_loss))
                    prev_loss = loss.data

                # Update the parameters of the meta optimizer
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

        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch,
                                                                                      final_loss / args.updates_per_epoch,
                                                                                      decrease_in_loss / args.updates_per_epoch))

    # Obtain the symbolic optimizer
    with torch.no_grad():
        weights = meta_optimizer.get_weights()
        expr = pretty_print.network(weights, funcs,
                                    ["m", "v", "g", "g2", "g3", "ag", "sg", "sm", "ad", "rs", "ld", "cd", "1",
                                     "2"], threshold=0.001)

    return expr


def eval_optimizer(expr, lr):
    model_sol = ConvNet()
    if args.cuda:
        model_sol.cuda()
    model_adam = copy.deepcopy(model_sol)
    if args.cuda:
        model_adam.cuda()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model_adam.parameters(), betas=(0.95, 0.95), lr=lr)

    expr_optimizer = ExprOptimizer(expr, exist_expr=[])
    expr_optimizer.reset(params=model_sol.parameters())
    acc_sol = expr_train(model=model_sol, expr_optimizer=expr_optimizer, criterion=criterion, num_epochs=100, lr=lr)

    acc_adam = train(model=model_adam, optimizer=optimizer, criterion=criterion, num_epochs=100, lr=lr)

    print("Accuracy by SOL:", acc_sol)
    print("Accuracy by Adam:", acc_adam)


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    lr = 1e-4
    expr = main(lr=lr)
    # expr = '-0.00225207*1 + 0.00148819*2 + 0.000921935*ad + 0.00162161*ag - 0.00104794*cd + 0.00267767*g - 0.00183933*g2 - 0.00146948*g3 - 0.00290243*ld + 0.00166183*rs + 0.00660202*sg + 0.00169124*sm + 0.00111049*v - 0.043213018357241*(0.0651711*1 + 0.0556003*2 - 0.0215169*ag + 0.0146631*cd + 0.11027*g2 - 0.0264856*m + 0.166236*rs + 0.192456*sg + 0.0936064*sm - 0.0396645*v)*(-0.120224*1 - 0.16459*2 - 0.0356411*ad + 0.16744*ag - 0.0596665*cd - 0.0179978*g - 0.0779471*g2 + 0.0448037*g3 - 0.0979423*ld - 0.0763634*m - 0.106966*rs - 0.0978009*sg - 0.0658281*sm + 0.0689286*v) + 0.00655705*(-0.316667*1 - 0.71541*2 - 0.18129*ad + 0.0528747*ag - 0.543897*cd - 0.0829313*g - 0.246089*g3 - 0.313546*ld + 0.127971*m - rs - 0.550116*sg - 0.306653*sm - 0.105331*v)**2 - 0.472772687673569*(0.0105729*1 - 0.00698664*2 - 0.00432824*ad - 0.00761302*ag + 0.00491977*cd - 0.012571*g + 0.00863517*g2 + 0.00689883*g3 + 0.0136261*ld - 0.00780184*rs - 0.0309947*sg - 0.0079399*sm - 0.00521343*v + 0.297170698642731*(0.0651711*1 + 0.0556003*2 - 0.0215169*ag + 0.0146631*cd + 0.11027*g2 - 0.0264856*m + 0.166236*rs + 0.192456*sg + 0.0936064*sm - 0.0396645*v)*(-0.120224*1 - 0.16459*2 - 0.0356411*ad + 0.16744*ag - 0.0596665*cd - 0.0179978*g - 0.0779471*g2 + 0.0448037*g3 - 0.0979423*ld - 0.0763634*m - 0.106966*rs - 0.0978009*sg - 0.0658281*sm + 0.0689286*v) - 0.00687591*(-0.316667*1 - 0.71541*2 - 0.18129*ad + 0.0528747*ag - 0.543897*cd - 0.0829313*g - 0.246089*g3 - 0.313546*ld + 0.127971*m - rs - 0.550116*sg - 0.306653*sm - 0.105331*v)**2 + 0.314278*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) + 0.111926 - 0.214104/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1))*(0.0385644*1 - 0.0254837*2 - 0.0157872*ad - 0.0277685*ag + 0.0179448*cd - 0.0458525*g + 0.0314967*g2 + 0.0251634*g3 + 0.0497012*ld - 0.0284571*rs - 0.113053*sg - 0.0289607*sm - 0.0190159*v + 0.163781940937042*(0.0651711*1 + 0.0556003*2 - 0.0215169*ag + 0.0146631*cd + 0.11027*g2 - 0.0264856*m + 0.166236*rs + 0.192456*sg + 0.0936064*sm - 0.0396645*v)*(-0.120224*1 - 0.16459*2 - 0.0356411*ad + 0.16744*ag - 0.0596665*cd - 0.0179978*g - 0.0779471*g2 + 0.0448037*g3 - 0.0979423*ld - 0.0763634*m - 0.106966*rs - 0.0978009*sg - 0.0658281*sm + 0.0689286*v) - 0.0832666425030771*exp(0.0283303*1 + 0.0680335*2 + 0.044158*ad - 0.0642208*ag - 0.0163111*cd + 0.0778092*g + 0.0631605*g2 + 0.0536437*g3 - 0.0432836*ld + 0.0146082*m + 0.145738*rs + 0.15691*sg + 0.114101*sm) - 0.461683*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) + 0.262024026922518 + 0.165085/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1)) - 0.100468931050517*(-0.115791718315353*1 + 0.0765162642644459*2 + 0.0474019467522985*ad + 0.0833762571796991*ag - 0.0538803445368844*cd + 0.137674591712402*g - 0.0945706214228778*g2 - 0.0755545263596478*g3 - 0.149230649690482*ld + 0.0854441051432725*rs + 0.339447787989981*sg + 0.0869561576830485*sm + 0.0570963985945727*v + 0.0913956860570681*(0.0651711*1 + 0.0556003*2 - 0.0215169*ag + 0.0146631*cd + 0.11027*g2 - 0.0264856*m + 0.166236*rs + 0.192456*sg + 0.0936064*sm - 0.0396645*v)*(-0.120224*1 - 0.16459*2 - 0.0356411*ad + 0.16744*ag - 0.0596665*cd - 0.0179978*g - 0.0779471*g2 + 0.0448037*g3 - 0.0979423*ld - 0.0763634*m - 0.106966*rs - 0.0978009*sg - 0.0658281*sm + 0.0689286*v) + 0.00187277611756505*(-0.316667*1 - 0.71541*2 - 0.18129*ad + 0.0528747*ag - 0.543897*cd - 0.0829313*g - 0.246089*g3 - 0.313546*ld + 0.127971*m - rs - 0.550116*sg - 0.306653*sm - 0.105331*v)**2 + 0.286300929537599*exp(0.0283303*1 + 0.0680335*2 + 0.044158*ad - 0.0642208*ag - 0.0163111*cd + 0.0778092*g + 0.0631605*g2 + 0.0536437*g3 - 0.0432836*ld + 0.0146082*m + 0.145738*rs + 0.15691*sg + 0.114101*sm) + 0.142481703433551*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) - 1 - 0.148808462866861/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1))**2 + 0.0635023863560535*exp(0.0283303*1 + 0.0680335*2 + 0.044158*ad - 0.0642208*ag - 0.0163111*cd + 0.0778092*g + 0.0631605*g2 + 0.0536437*g3 - 0.0432836*ld + 0.0146082*m + 0.145738*rs + 0.15691*sg + 0.114101*sm) + 0.186105928113977*exp(-0.0426706*1 + 0.0281971*2 + 0.0174682*ad + 0.0307251*ag - 0.0198555*cd + 0.0507346*g - 0.0348503*g2 - 0.0278427*g3 - 0.0549932*ld + 0.0314871*rs + 0.12509*sg + 0.0320443*sm + 0.0210407*v - 0.598920345306396*(0.0651711*1 + 0.0556003*2 - 0.0215169*ag + 0.0146631*cd + 0.11027*g2 - 0.0264856*m + 0.166236*rs + 0.192456*sg + 0.0936064*sm - 0.0396645*v)*(-0.120224*1 - 0.16459*2 - 0.0356411*ad + 0.16744*ag - 0.0596665*cd - 0.0179978*g - 0.0779471*g2 + 0.0448037*g3 - 0.0979423*ld - 0.0763634*m - 0.106966*rs - 0.0978009*sg - 0.0658281*sm + 0.0689286*v) + 0.00436663*(-0.316667*1 - 0.71541*2 - 0.18129*ad + 0.0528747*ag - 0.543897*cd - 0.0829313*g - 0.246089*g3 - 0.313546*ld + 0.127971*m - rs - 0.550116*sg - 0.306653*sm - 0.105331*v)**2 + 0.0986908041916254*exp(0.0283303*1 + 0.0680335*2 + 0.044158*ad - 0.0642208*ag - 0.0163111*cd + 0.0778092*g + 0.0631605*g2 + 0.0536437*g3 - 0.0432836*ld + 0.0146082*m + 0.145738*rs + 0.15691*sg + 0.114101*sm) - 0.536157*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) + 0.28095/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1)) - 0.00718121*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) + 0.281208*sign(0.00319707*1 - 0.00211266*2 - 0.00130879*ad - 0.00230207*ag + 0.00148767*cd - 0.00380127*g + 0.00261115*g2 + 0.0020861*g3 + 0.00412034*ld - 0.00235916*rs - 0.00937235*sg - 0.00240091*sm - 0.00157646*v - 0.0124156*(-0.316667*1 - 0.71541*2 - 0.18129*ad + 0.0528747*ag - 0.543897*cd - 0.0829313*g - 0.246089*g3 - 0.313546*ld + 0.127971*m - rs - 0.550116*sg - 0.306653*sm - 0.105331*v)**2 + 0.430474*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) - 0.0495119/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1)) - 0.180486147012986 - 0.646377/(4.38667458007599*exp(-1.63949*1 + 1.08339*2 + 0.671161*ad + 1.18052*ag - 0.762888*cd + 1.94932*g - 1.33902*g2 - 1.06977*g3 - 2.11295*ld + 1.2098*rs + 4.80622*sg + 1.23121*sm + 0.808424*v - 8.31490457057953*(0.0651711*1 + 0.0556003*2 - 0.0215169*ag + 0.0146631*cd + 0.11027*g2 - 0.0264856*m + 0.166236*rs + 0.192456*sg + 0.0936064*sm - 0.0396645*v)*(-0.120224*1 - 0.16459*2 - 0.0356411*ad + 0.16744*ag - 0.0596665*cd - 0.0179978*g - 0.0779471*g2 + 0.0448037*g3 - 0.0979423*ld - 0.0763634*m - 0.106966*rs - 0.0978009*sg - 0.0658281*sm + 0.0689286*v) + 0.335987*(-0.316667*1 - 0.71541*2 - 0.18129*ad + 0.0528747*ag - 0.543897*cd - 0.0829313*g - 0.246089*g3 - 0.313546*ld + 0.127971*m - rs - 0.550116*sg - 0.306653*sm - 0.105331*v)**2 + 1.29992321488984*exp(0.0283303*1 + 0.0680335*2 + 0.044158*ad - 0.0642208*ag - 0.0163111*cd + 0.0778092*g + 0.0631605*g2 + 0.0536437*g3 - 0.0432836*ld + 0.0146082*m + 0.145738*rs + 0.15691*sg + 0.114101*sm) + 4.1616*sign(0.0133245*ad + 0.0154127*g2 + 0.0103594*m - 0.0745614*sm) + 3.0343/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1)) + 1) + 0.0163605/(exp(0.91169*2 + 0.654391*ad + 0.494202*cd - 1.7263*g + 1.76861*g2 + 0.379465*g3 - 0.380043*ld - 1.62262*rs - 2.70369*sg - 3.52032*sm - 1.04011*v) + 1)'
    eval_optimizer(expr, lr=lr)
