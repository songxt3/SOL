import copy
import time
from itertools import cycle
import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import functions
import pretty_print
from meta_optimizer import SymbolicOptimizer, MetaModel, ExprOptimizer
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.autograd import Variable
from GCNConv import GCNConv
import torch.nn as nn
from torch_geometric.nn import JumpingKnowledge
import math

torch.manual_seed(228)

DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# 创建GAT模型
class GATModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class JKNet(nn.Module):
    def __init__(self, dataset, mode='max', num_layers=6, hidden=16):
        super(JKNet, self).__init__()
        self.num_layers = num_layers
        self.mode = mode

        self.conv0 = GCNConv(dataset.num_node_features, hidden)
        self.dropout0 = nn.Dropout(p=0.5)

        for i in range(1, self.num_layers):
            setattr(self, 'conv{}'.format(i), GCNConv(hidden, hidden))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=0.5))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden, dataset.num_classes)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        lin_times = 0
        mes_times = 0
        aggr_times = 0
        up_times = 0
        jk_times = 0

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x, linear_time, message_time, aggregate_time, update_time = conv(x, edge_index)
            lin_times += linear_time
            mes_times += message_time
            aggr_times += aggregate_time
            up_times += update_time

            x = dropout(F.relu(x))
            layer_out.append(x)

        start_time = time.time()
        h = self.jk(layer_out)  # JK层
        end_time = time.time()
        jk_times = end_time - start_time

        h = self.fc(h)
        h = F.log_softmax(h, dim=1)

        return h, lin_times, mes_times, aggr_times, up_times, jk_times


def main(learning_rate):
    dataset = Planetoid(root='data/CiteSeer', name='CiteSeer')
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    train_iter = cycle(loader)

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    hidden_dim = 8
    num_heads = 8
    meta_model = GATModel(num_features, num_classes, hidden_dim, num_heads)
    # meta_model = JKNet(dataset, mode='max')
    meta_model.to(DEVICE)

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

    meta_optimizer.to(DEVICE)

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)

    criterion = torch.nn.NLLLoss()
    start_time = time.time()
    for epoch in range(50):
        decrease_in_loss = 0.0
        final_loss = 0.0
        for i in range(10):
            model = GATModel(num_features, num_classes, hidden_dim, num_heads)
            # model = JKNet(dataset, mode='max')
            model.to(DEVICE)

            data = next(train_iter)
            data.to(DEVICE)
            output = model(data.x, data.edge_index)
            # output, _, _, _, _, _ = model(data)
            initial_loss = criterion(output[data.train_mask], data.y[data.train_mask])

            length = 250
            unrolled_len = 50
            for k in range(int(length // unrolled_len)):
                meta_optimizer.reset(keep_states=k > 0, model=model, params=model.parameters())

                loss_sum = 0
                prev_loss = torch.zeros(1)
                prev_loss = prev_loss.to(DEVICE)
                for j in range(unrolled_len):
                    data = next(train_iter)
                    f_x = model(data.x, data.edge_index)
                    # f_x, _, _, _, _, _ = model(data)
                    loss = criterion(f_x[data.train_mask], data.y[data.train_mask])
                    model.zero_grad()
                    loss.backward()
                    # 梯度裁剪，避免出现梯度爆炸情况
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    meta_model = meta_optimizer.meta_update(model, step=(k * unrolled_len + j + 1), T=length,
                                                            lr=learning_rate)

                    f_x = meta_model(data.x, data.edge_index)
                    # f_x, _, _, _, _, _ = meta_model(data)
                    loss = criterion(f_x[data.train_mask], data.y[data.train_mask])

                    # loss_sum += (loss - Variable(prev_loss))
                    loss_sum += loss

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
    end_time = time.time()
    print("\n searching time:", end_time - start_time)
    # test process
    with torch.no_grad():
        weights = meta_optimizer.get_weights()
        expr = pretty_print.network(weights, funcs,
                                    ["m", "v", "g", "g2", "g3", "ag", "sg", "sm", "ad", "rs", "ld", "cd", "1",
                                     "2"])

    expr_optimizer = ExprOptimizer(expr)

    test_model = GATModel(num_features, num_classes, hidden_dim, num_heads)
    # test_model = JKNet(dataset, mode='max')

    eval_optimizer(test_model, expr_optimizer, num_epochs=500, lr=learning_rate)


def cosine_annealing_lr(epoch, T_max, eta_min, eta_max):
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * (epoch % T_max) / T_max))


def eval_optimizer(model, expr_optimizer, num_epochs, lr):
    # load data
    dataset = Planetoid(root='data/CiteSeer', name='CiteSeer')
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    copied_model = copy.deepcopy(model)
    copied_model.to(DEVICE)
    model.to(DEVICE)

    criterion = torch.nn.NLLLoss()

    # expr train
    model.train()
    expr_optimizer.reset(params=model.parameters())
    for epoch in range(num_epochs):
        for data in loader:
            data.cuda()
            model.zero_grad()
            output = model(data.x, data.edge_index)
            # output, _, _, _, _, _ = model(data)
            loss = criterion(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            # 梯度裁剪，避免出现梯度爆炸情况
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            expr_optimizer.meta_update(model, epoch + 1, num_epochs, lr)

    model.eval()
    correct = 0
    for data in loader:
        data.cuda()
        out = model(data.x, data.edge_index)
        # out, _, _, _, _, _ = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    total = int(data.test_mask.sum())
    accuracy = correct / total
    print('expr acc:', accuracy)

    # adam train
    copied_model.train()
    optimizer = torch.optim.Adam(copied_model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for data in loader:
            data.cuda()
            optimizer.zero_grad()
            output = copied_model(data.x, data.edge_index)
            # output, _, _, _, _, _ = copied_model(data)
            loss = criterion(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

    copied_model.eval()
    correct = 0
    for data in loader:
        data.cuda()
        out = copied_model(data.x, data.edge_index)
        # out, _, _, _, _, _ = copied_model(data)
        pred = out.argmax(dim=1)
        correct += int((pred[data.test_mask] == data.y[data.test_mask]).sum())
    total = int(data.test_mask.sum())
    accuracy = correct / total
    print('\n Adam acc:', accuracy)


if __name__ == "__main__":
    main(learning_rate=5e-4)
    '''
    dataset = Planetoid(root='data/Cora', name='Cora')

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes
    hidden_dim = 8
    num_heads = 8
    expr = '-0.0530499*1 - 0.0371871*2 + 0.252877*ad + 0.0933372*ag + 0.097649*g - 0.0200291*g2 + 0.038608*g3 - 0.0759638*ld - 0.024732*m + 0.10302*rs + 0.195674*sg + 0.121787*sm - 0.05104*v - 0.629146635532379*(0.0886567*1 + 0.17968*2 - 0.0290982*ad - 0.122808*ag + 0.0525911*cd + 0.0842566*g - 0.0297855*g2 - 0.0821611*g3 + 0.0502028*ld + 0.0657626*m + 0.0260629*rs - 0.0524304*sg + 0.0183726*v)*(-0.0610307*1 + 0.0693926*2 - 0.141194*ad - 0.0221373*ag + 0.0204287*cd - 0.0308087*g - 0.0470189*g2 - 0.0449847*g3 - 0.0572929*ld - 0.0413996*m - 0.0829919*rs - 0.175536*sg - 0.253241*sm - 0.05186*v) - 0.0421611*(-1 - 0.777445*2 + 0.92263*ad + 0.266501*ag - 0.434619*cd + 0.269968*g + 0.216886*g2 - 0.107786*g3 - 0.460876*ld - 0.0815775*m + 0.507854*rs + 0.66153*sg + 0.447739*sm - 0.435059*v)**2 + 0.757200120848698*exp(0.0110474*1 + 0.225613*ad + 0.0274144*ag - 0.0638586*cd + 0.0355755*g + 0.0268286*g2 + 0.0296286*g3 + 0.0506268*m - 0.0202512*rs + 0.139274*sg + 0.0635237*sm + 0.13368*v) + 0.353733*sign(-0.0253657*g - 0.0419167*g3 + 0.0338811*ld - 0.030724*m) + 0.152280391293483 - 0.755975/(exp(-2.47546*1 - 0.648819*2 + 1.52928*ad + 1.31037*ag - 2.63901*cd - 1.24421*g + 1.65286*g2 - 2.7346*g3 - 2.31106*ld + 3.16445*m + 1.20386*rs + 1.51008*sg + 0.769659*sm + 0.658572*v) + 1)'
    expr_optimizer = ExprOptimizer(expr)

    test_model = GATModel(num_features, num_classes, hidden_dim, num_heads)
    # test_model = JKNet(dataset, mode='max')

    test_optimizer(test_model, expr_optimizer, num_epochs=500, lr=0.0005)
    '''
