import random
import math
import torch
import torchvision.transforms as transforms
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# from treefromexpr import get_tree
import numpy as np
from itertools import cycle
from torch.autograd import Variable
import torch.nn.functional as F

SR_configs = {
    "task": {
        # Deep Symbolic Regression
        "task_type": "regression",

        "function_set": ["add", "sub", "mul", "div", "exp", "log", "sqrt", "n2", "tanh", "const", "poly"],

        "metric": "inv_nrmse",
        "metric_params": [1.0],

        "extra_metric_test": None,
        "extra_metric_test_params": [],

        "threshold": 1e-12,

        "protected": False,

        "reward_noise": 0.0,
        "reward_noise_type": "r",
        "normalize_variance": False,

        "decision_tree_threshold_set": [],

        "poly_optimizer_params": {
            "degree": 3,
            "coef_tol": 1e-6,
            "regressor": "dso_least_squares",
            "regressor_params": {
                "cutoff_p_value": 1.0,
                "n_max_terms": None,
                "coef_tol": 1e-6
            }
        }
    },

    "gp_meld": {
        "run_gp_meld": True,
        "population_size": 100,
        "generations": 20,
        "crossover_operator": "cxOnePoint",
        "p_crossover": 0.5,
        "mutation_operator": "multi_mutate",
        "p_mutate": 0.5,
        "tournament_size": 5,
        "train_n": 50,
        "mutate_tree_max": 3,
        "verbose": False,
        "parallel_eval": True
    },

    "training": {
        "n_samples": 20000,
        "batch_size": 500,
        "epsilon": 0.02,
        "n_cores_batch": 2
    },

    "controller": {
        "learning_rate": 0.0025,
        "entropy_weight": 0.03,
        "entropy_gamma": 0.7,
        "pqt": True,
        "pqt_k": 10,
        "pqt_batch_size": 1,
        "pqt_weight": 200.0,
        "pqt_use_pg": False
    },

    "prior": {
        "length": {
            "min_": 4,
            "max_": 100,
            "on": True
        },
        "inverse": {
            "on": True
        },
        "trig": {
            "on": True
        },
        "const": {
            "on": True
        },
        "no_inputs": {
            "on": True
        },
        "uniform_arity": {
            "on": True
        },
        "soft_length": {
            "loc": 10,
            "scale": 5,
            "on": True
        },
        "domain_range": {
            "on": True
        }
    }
}


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    # x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return x1


def expr2dst(expr, opera_lst, fea_lst):
    tree = get_tree(expr)

    # idx_lst & node_matrix
    opera_idx_lst = []
    fea_idx_lst = []
    global nodeID
    nodeID = 0

    def visit(node):
        global nodeID
        if node is not None:
            if node.node_type == 2:
                opera_idx_lst.append(opera_lst.index(node.op))
            elif node.node_type == 1:
                opera_idx_lst.append(opera_lst.index(node.op))
            else:
                fea_idx_lst.append(fea_lst.index(node.op))
            node.nodeID = nodeID
            nodeID += 1
        if node.left is not None:
            visit(node.left)
        if node.right is not None:
            visit(node.right)

    visit(tree)
    N_opera = len(opera_idx_lst)
    N_fea = len(fea_idx_lst)
    L_opera = len(opera_lst)
    L_fea = len(fea_lst)
    node_mat_opera = np.zeros((N_opera, L_opera))
    for i, v in enumerate(opera_idx_lst):
        node_mat_opera[i, v] = 10
    node_mat_fea = np.zeros((N_fea, L_fea))
    for i, v in enumerate(fea_idx_lst):
        node_mat_fea[i, v] = 10

    return tree, node_mat_opera, node_mat_fea


class Cifar_f:
    def __init__(self, cifarAB='A', shuffle_cifar=True, num_workers=2, batch_size=None, pin_memory=None,
                 **kwargs):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(
            root='../datasets', train=True, download=True, transform=transform_train
        )
        subset_indices = list(range(50000))  # Choose the first 5000 samples
        subset_sampler = SubsetRandomSampler(subset_indices)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,
                                                       num_workers=num_workers, sampler=subset_sampler)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='../datasets', train=False, download=True, transform=transform_test
        )
        self.testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )


def model2expr(model, inputs):
    outputs = model(inputs)
    inputs_array = inputs.cpu().detach().numpy()
    outputs_array = outputs.cpu().detach().numpy()

    sample_num = 1000
    sample_list = [i for i in range(len(inputs_array))]
    sample_list = random.sample(sample_list, sample_num)
    data = inputs_array[sample_list, :]
    label = outputs_array[sample_list]

    SR_model = const_main.train(start_expr=['x0 / x1'],
                                num_gen=10,
                                num_epoch=1000,
                                train_d=data,
                                train_t=label,
                                test_d=data,
                                test_t=label,
                                num_fea=13,
                                primitive_lst=['+', '-', '*', '/', 'sqrt', 'pow', 'exp', 'log', 'tanh',
                                               'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                                               'x8', 'x9', 'x10', 'x11', 'x12'])

    # SR_model.fit(data, label)

    return SR_model


def additional_l2o(max_epoch, data_loader, updates_per_epoch, Model, optimizer_steps, truncated_bptt_step, meta_optimizer, optimizer, criterion, scheduler=None, use_cuda=True, learning_rate=0.001):
    for epoch in range(max_epoch):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = cycle(data_loader)
        for i in range(updates_per_epoch):
            # Sample a new model
            model = Model()
            if use_cuda:
                model.cuda()

            x, y = next(train_iter)
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # Compute initial loss of the model
            f_x = model(x)
            # initial_loss = F.nll_loss(f_x, y)
            initial_loss = criterion(f_x, y)

            for k in range(optimizer_steps // truncated_bptt_step):
                # 2
                # Keep states for truncated BPTT
                meta_optimizer.reset(keep_states=k > 0, model=model, params=model.parameters())

                loss_sum = 0
                prev_loss = torch.zeros(1)
                if use_cuda:
                    prev_loss = prev_loss.cuda()
                for j in range(truncated_bptt_step):
                    x, y = next(train_iter)
                    if use_cuda:
                        x, y = x.cuda(), y.cuda()
                    x, y = Variable(x), Variable(y)

                    # First we need to compute the gradients of the model
                    f_x = model(x)
                    # loss = F.nll_loss(f_x, y)
                    loss = criterion(f_x, y)
                    if torch.isnan(loss):
                        continue
                    model.zero_grad()
                    loss.backward()

                    # Perfom a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    # 3
                    meta_model = meta_optimizer.meta_update(model, step=(k * truncated_bptt_step + j + 1),
                                                            T=optimizer_steps, lr=learning_rate)

                    # Compute a loss for a step the meta optimizer
                    f_x = meta_model(x)
                    # loss = F.nll_loss(f_x, y)
                    loss = criterion(f_x, y)

                    loss_sum += (loss - Variable(prev_loss))

                    prev_loss = loss.data

                # Update the parameters of the meta optimizer
                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

            # Compute relative decrease in the loss function w.r.t initial
            # value
            decrease_in_loss += loss.item() / initial_loss.item()
            final_loss += loss.item()

        if scheduler:
            scheduler.step()
        print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch,
                                                                                      final_loss / updates_per_epoch,
                                                                                      decrease_in_loss / updates_per_epoch))

    return meta_optimizer


