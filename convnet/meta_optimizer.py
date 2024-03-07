from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import preprocess_gradients
from model import BasicBlock
# from treefromexpr import BinOp
# from operations import *
import functions
from torch.optim import Optimizer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# DEVICE = torch.device('cpu')


def detach_var(v):
    var = Variable(v.data, requires_grad=True).to(DEVICE)
    var.retain_grad()
    return var


class SimpleSymbolicOptimizer(nn.Module):
    def __init__(self, model, n_layers, in_dim=1, funcs=None, initial_weights=None, beta1=0.9, beta2=0.999,
                 exist_expr=None):
        super(SimpleSymbolicOptimizer, self).__init__()
        self.meta_model = model
        self.beta1 = torch.tensor(beta1).type(torch.float)
        self.beta2 = torch.tensor(beta2).type(torch.float)
        self.exist_expr = []
        if exist_expr is not None:
            for item in exist_expr:
                item = str(item)
                print(item)
                item = item.replace('exp', 'torch.exp')
                item = item.replace('sin', 'torch.sin')
                item = item.replace('sign', 'torch.sign')
                self.exist_expr.append(item)

        self.depth = n_layers
        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i],
                                      in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        self.hidden_layers = nn.Sequential(*layers)

    def cuda(self):
        super(SimpleSymbolicOptimizer, self).cuda()
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].cuda()

    def reset(self, keep_states, model, params):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if not keep_states:
            mt = []
            vt = []
            for i, p in enumerate(params):
                N_this = np.prod(p.size())
                mt.append(torch.zeros(N_this, 1).to(DEVICE))
                vt.append(torch.zeros(N_this, 1).to(DEVICE))

            self.mt_all = torch.cat(mt)
            self.vt_all = torch.cat(vt)

    def forward(self, input, sample=True, reuse_u=False):
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)

        h = torch.matmul(h, self.output_weight)
        return h

    def build_fea(self, grads, step, T):
        g = grads
        g2 = torch.pow(grads, 2)
        g3 = torch.pow(grads, 3)
        temp = preprocess_gradients(grads)
        ag = temp[:, 0].unsqueeze(-1)
        self.mt_all = self.beta1 * self.mt_all + (1.0 - self.beta1) * grads
        m = self.mt_all / (1 - torch.pow(self.beta1, step))
        self.vt_all = self.beta2 * self.vt_all + (1.0 - self.beta2) * g2
        vt_hat = self.vt_all / (1 - torch.pow(self.beta2, step))
        v = torch.sqrt(vt_hat) + 1e-12
        sg = torch.sign(g)
        sm = torch.sign(m)
        ad = m / v
        rs = g / v
        fea_array = [m, v, g, g2, g3, ag, sg, sm, ad, rs,
                     torch.tensor(1).type(torch.float).view([1, 1]).to(DEVICE),
                     torch.tensor(2).type(torch.float).view([1, 1]).to(DEVICE)]
        temp_expr = self.exist_expr
        if temp_expr:
            str1 = temp_expr[0]
            k1 = eval(str1)
            fea_array.append(k1)
        if temp_expr[1:]:
            str2 = temp_expr[1]
            k2 = eval(str2)
            fea_array.append(k2)
        return fea_array

    def meta_update(self, model_with_grads, step, T, lr):
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))

        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)

        fea_array = self.build_fea(flat_grads, step, T)
        for i in range(len(fea_array)):
            if fea_array[i].size(0) != flat_grads.size(0):
                fea_array[i] = fea_array[i].expand_as(flat_grads)
        inputs = Variable(torch.cat(fea_array, 1))

        if torch.isnan(inputs).any():
            print('inputs nan occurred')
            inputs = torch.where(torch.isnan(inputs), torch.full_like(inputs, 0), inputs)
        updates = self(inputs, sample=True, reuse_u=False)

        if torch.isnan(updates).any():
            updates = torch.where(torch.isnan(updates), torch.full_like(updates, 0), updates)

            print('updates nan occurred')

        updated_params = flat_params - lr * updates

        self.meta_model.set_flat_params(updated_params)
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

    def get_weights(self):
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

    def move_to_cpu(self):
        self.meta_model.model = self.meta_model.model.to('cpu')
        self.mt_all = self.mt_all.to('cpu')
        self.vt_all = self.vt_all.to('cpu')
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i] = self.hidden_layers[i].to('cpu')
            self.hidden_layers[i].eps = self.hidden_layers[i].eps.to('cpu')
        self.output_weight = nn.Parameter(self.output_weight.to('cpu'))


class SymbolicOptimizer(nn.Module):
    def __init__(self, model, n_layers, in_dim=1, funcs=None, initial_weights=None, beta1=0.95, beta2=0.95,
                 exist_expr=None):
        super(SymbolicOptimizer, self).__init__()
        self.meta_model = model
        self.beta1 = torch.tensor(beta1).type(torch.float)
        self.beta2 = torch.tensor(beta2).type(torch.float)
        self.exist_expr = []
        if exist_expr is not None:
            for item in exist_expr:
                item = str(item)
                print(item)
                item = item.replace('exp', 'torch.exp')
                item = item.replace('sin', 'torch.sin')
                item = item.replace('sign', 'torch.sign')
                self.exist_expr.append(item)

        self.depth = n_layers
        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i],
                                      in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        self.hidden_layers = nn.Sequential(*layers)

    def cuda(self):
        super(SymbolicOptimizer, self).cuda()
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i].cuda()

    def reset(self, keep_states, model, params):
        self.meta_model.reset()
        self.meta_model.copy_params_from(model)
        if not keep_states:
            mt = []
            vt = []
            for i, p in enumerate(params):
                N_this = np.prod(p.size())
                mt.append(torch.zeros(N_this, 1).to(DEVICE))
                vt.append(torch.zeros(N_this, 1).to(DEVICE))

            self.mt_all = torch.cat(mt)
            self.vt_all = torch.cat(vt)

    def forward(self, input, sample=True, reuse_u=False):
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)

        h = torch.matmul(h, self.output_weight)
        return h

    def build_fea(self, grads, step, T):
        g = grads
        g2 = torch.pow(grads, 2)
        g3 = torch.pow(grads, 3)
        temp = preprocess_gradients(grads)
        ag = temp[:, 0].unsqueeze(-1)
        self.mt_all = self.beta1 * self.mt_all + (1.0 - self.beta1) * grads
        m = self.mt_all / (1 - torch.pow(self.beta1, step))
        self.vt_all = self.beta2 * self.vt_all + (1.0 - self.beta2) * g2
        vt_hat = self.vt_all / (1 - torch.pow(self.beta2, step))
        v = torch.sqrt(vt_hat) + 1e-12
        sg = torch.sign(g)
        sm = torch.sign(m)
        ad = m / v
        rs = g / v
        ld = torch.tensor(1 - step / T).type(torch.float).view([1, 1]).to(DEVICE)
        cd = 0.5 * (1 + torch.cos(3.14 * torch.tensor(step / T).type(torch.float))).view([1, 1]).to(DEVICE)
        fea_array = [m, v, g, g2, g3, ag, sg, sm, ad, rs, ld, cd,
                     torch.tensor(1).type(torch.float).view([1, 1]).to(DEVICE),
                     torch.tensor(2).type(torch.float).view([1, 1]).to(DEVICE)]
        # fea_array = [m, v, g, g2, g3, ag, sg, sm, ad, rs, torch.tensor(1).type(torch.float).view([1, 1]).to(DEVICE),
        #              torch.tensor(2).type(torch.float).view([1, 1]).to(DEVICE)]
        temp_expr = self.exist_expr
        if temp_expr:
            str1 = temp_expr[0]
            k1 = eval(str1)
            fea_array.append(k1)
        if temp_expr[1:]:
            str2 = temp_expr[1]
            k2 = eval(str2)
            fea_array.append(k2)
        return fea_array

    def meta_update(self, model_with_grads, step, T, lr):
        grads = []

        for module in model_with_grads.children():
            grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
            grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))

        flat_params = self.meta_model.get_flat_params().unsqueeze(-1)
        flat_grads = torch.cat(grads)

        fea_array = self.build_fea(flat_grads, step, T)
        for i in range(len(fea_array)):
            if fea_array[i].size(0) != flat_grads.size(0):
                fea_array[i] = fea_array[i].expand_as(flat_grads)
        inputs = Variable(torch.cat(fea_array, 1))

        if torch.isnan(inputs).any():
            print('inputs nan occurred')
            inputs = torch.where(torch.isnan(inputs), torch.full_like(inputs, 0), inputs)
        updates = self(inputs, sample=True, reuse_u=False)

        if torch.isnan(updates).any():
            updates = torch.where(torch.isnan(updates), torch.full_like(updates, 0), updates)

            print('updates nan occurred')

        updated_params = flat_params - lr * updates

        self.meta_model.set_flat_params(updated_params)
        self.meta_model.copy_params_to(model_with_grads)
        return self.meta_model.model

    def get_weights(self):
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

    def move_to_cpu(self):
        self.meta_model.model = self.meta_model.model.to('cpu')
        self.mt_all = self.mt_all.to('cpu')
        self.vt_all = self.vt_all.to('cpu')
        for i in range(len(self.hidden_layers)):
            self.hidden_layers[i] = self.hidden_layers[i].to('cpu')
            self.hidden_layers[i].eps = self.hidden_layers[i].eps.to('cpu')
        self.output_weight = nn.Parameter(self.output_weight.to('cpu'))


class ExprOptimizer(nn.Module):
    def __init__(self, expr, beta1=0.95, beta2=0.95, exist_expr=None):
        super(ExprOptimizer, self).__init__()
        self.expr = str(expr)
        self.expr = self.expr.replace('exp', 'torch.exp')
        self.expr = self.expr.replace('sin', 'torch.sin')
        # self.expr = self.expr.replace('sign', 'torch.sign')
        print(expr)
        self.beta1 = torch.tensor(beta1).type(torch.float)
        self.beta2 = torch.tensor(beta2).type(torch.float)
        self.exist_expr = []
        if exist_expr is not None:
            for item in exist_expr:
                item = str(item)
                print(item)
                item = item.replace('exp', 'torch.exp')
                item = item.replace('sin', 'torch.sin')
                item = item.replace('sign', 'torch.sign')
                self.exist_expr.append(item)

    def reset(self, params):
        mt = []
        vt = []
        for i, p in enumerate(params):
            N_this = np.prod(p.size())
            mt.append(torch.zeros(N_this, 1).to(DEVICE))
            vt.append(torch.zeros(N_this, 1).to(DEVICE))

        self.mt_all = torch.cat(mt)
        self.vt_all = torch.cat(vt)

    def build_fea(self, grads, step, T):
        g = grads
        g2 = torch.pow(grads, 2)
        g3 = torch.pow(grads, 3)
        temp = preprocess_gradients(grads)
        ag = temp[:, 0].unsqueeze(-1)
        self.mt_all = self.beta1 * self.mt_all + (1.0 - self.beta1) * grads
        m = self.mt_all / (1 - torch.pow(self.beta1, step))
        self.vt_all = self.beta2 * self.vt_all + (1.0 - self.beta2) * g2
        vt_hat = self.vt_all / (1 - torch.pow(self.beta2, step))
        v = torch.sqrt(vt_hat) + 1e-12
        sg = torch.sign(g)
        sm = torch.sign(m)
        ad = m / v
        rs = g / v
        ld = torch.tensor(1 - step / T).type(torch.float).view([1, 1]).to(DEVICE)
        cd = 0.5 * (1 + torch.cos(3.14 * torch.tensor(step / T).type(torch.float))).view([1, 1]).to(DEVICE)
        fea_array = [m, v, g, g2, g3, ag, sg, sm, ad, rs, ld, cd,
                     torch.tensor(1).type(torch.float).view([1, 1]).to(DEVICE),
                     torch.tensor(2).type(torch.float).view([1, 1]).to(DEVICE)]
        temp_expr = self.exist_expr
        if temp_expr:
            str1 = temp_expr[0]
            k1 = eval(str1)
            fea_array.append(k1)
        if temp_expr[1:]:
            str2 = temp_expr[1]
            k2 = eval(str2)
            fea_array.append(k2)
        return fea_array

    def meta_update(self, model_with_grads, step, T, lr):
        grads = []
        params = []

        def get_grads(model):
            for module in model.children():
                if module._parameters and module._parameters['weight'] is not None:
                    grads.append(module._parameters['weight'].grad.data.view(-1).unsqueeze(-1))
                    params.append(module._parameters['weight'].data.view(-1).unsqueeze(-1))
                if module._parameters and module._parameters['bias'] is not None:
                    grads.append(module._parameters['bias'].grad.data.view(-1).unsqueeze(-1))
                    params.append(module._parameters['bias'].data.view(-1).unsqueeze(-1))
                if type(module) == nn.Sequential or type(module) == BasicBlock:
                    get_grads(module)

        get_grads(model_with_grads)

        flat_grads = torch.cat(grads)
        flat_params = torch.cat(params)

        fea_array = self.build_fea(flat_grads, step, T)
        for i in range(len(fea_array)):
            if fea_array[i].size(0) != flat_grads.size(0):
                fea_array[i] = fea_array[i].expand_as(flat_grads)
        inputs = Variable(torch.cat(fea_array, 1))

        updates = self(inputs)
        if torch.isnan(updates).any():
            updated_params = flat_params
        else:
            updated_params = flat_params - lr * updates

        offset = 0

        def set_parameters(model, offset):
            for module in model.children():
                if module._parameters and module._parameters['weight'] is not None:
                    weight_shape = module._parameters['weight'].size()
                    weight_flat_size = reduce(mul, weight_shape, 1)
                    module._parameters['weight'].data = updated_params[offset:offset + weight_flat_size].view(
                        *weight_shape)
                    offset += weight_flat_size
                if module._parameters and module._parameters['bias'] is not None:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'].data = updated_params[offset:offset + bias_flat_size].view(*bias_shape)
                    offset += bias_flat_size
                if type(module) == nn.Sequential or type(module) == BasicBlock:
                    set_parameters(module, offset)

        set_parameters(model_with_grads, offset)

    def forward(self, inputs):
        m = inputs[:, 0]
        v = inputs[:, 1]
        g = inputs[:, 2]
        g2 = inputs[:, 3]
        g3 = inputs[:, 4]
        ag = inputs[:, 5]
        sg = inputs[:, 6]
        sm = inputs[:, 7]
        ad = inputs[:, 8]
        rs = inputs[:, 9]
        ld = inputs[:, 10]
        cd = inputs[:, 11]
        if len(self.exist_expr) == 1:
            k1 = inputs[:, 14]
        if len(self.exist_expr) == 2:
            k1 = inputs[:, 14]
            k2 = inputs[:, 15]

        def sign(x):
            if torch.is_tensor(x):
                return torch.sign(x)
            else:
                return np.sign(x)

        updates = eval(self.expr)
        return updates.unsqueeze(-1)


class Solo(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, expr='ad'):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Solo, self).__init__(params, defaults)

        self.base_lrs = list(map(lambda group: group['lr'], self.param_groups))
        self.expr = str(expr)
        print(expr)
        self.expr = self.expr.replace('exp', 'torch.exp')

    def __setstate__(self, state):
        super(Solo, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                fea_array = self.build_fea(grad, state['step'], exp_avg, exp_avg_sq, beta1, beta2)
                for i in range(len(fea_array)):
                    if fea_array[i].size(0) != grad.size(0):
                        fea_array[i] = fea_array[i].expand_as(grad)
                inputs = Variable(torch.stack(fea_array, 0))

                updates = self.compute_updates(inputs)
                step_size = group['lr'] * updates

                p.data.add_(-step_size)

        return loss

    def build_fea(self, grad, step, mt, vt, beta1, beta2):
        g = grad
        g2 = torch.pow(grad, 2)
        g3 = torch.pow(grad, 3)
        temp = preprocess_gradients(grad)
        ag = temp
        m = mt / (1 - beta1 ** step)
        vt_hat = vt / (1 - beta2 ** step)
        v = torch.sqrt(vt_hat) + 1e-12
        sg = torch.sign(g)
        sm = torch.sign(m)
        ad = m / v
        rs = g / v

        fea_array = [m, v, g, g2, g3, ag, sg, sm, ad, rs, torch.tensor(1).type(torch.float).view([1]).to(DEVICE),
                     torch.tensor(2).type(torch.float).view([1]).to(DEVICE)]
        return fea_array

    def compute_updates(self, inputs):
        m = inputs[0, :]
        v = inputs[1, :]
        g = inputs[2, :]
        g2 = inputs[3, :]
        g3 = inputs[4, :]
        ag = inputs[5, :]
        sg = inputs[6, :]
        sm = inputs[7, :]
        ad = inputs[8, :]
        rs = inputs[9, :]

        def sign(x):
            if torch.is_tensor(x):
                return torch.sign(x)
            else:
                return np.sign(x)

        updates = eval(self.expr)
        return updates


class SymbolicLayer(nn.Module):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""

    def __init__(self, funcs=None, initial_weight=None, init_stddev=0.1, in_dim=None):
        """
        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """
        super().__init__()

        if funcs is None:
            funcs = functions.default_func
        self.initial_weight = initial_weight
        self.W = None  # Weight matrix
        self.built = False  # Boolean whether weights have been initialized

        self.output = None  # tensor for layer output
        self.n_funcs = len(funcs)  # Number of activation functions (and number of layer outputs)
        self.funcs = [func.torch for func in funcs]  # Convert functions to list of PyTorch functions
        self.n_double = functions.count_double(funcs)  # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double  # Number of activation functions that take 1 input

        self.out_dim = self.n_funcs + self.n_double

        if self.initial_weight is not None:  # use the given initial weight
            self.W = nn.Parameter(self.initial_weight.clone().detach())  # copies
            self.built = True
        else:
            self.W = torch.normal(mean=0.0, std=init_stddev, size=(in_dim, self.out_dim))

    def forward(self, x):  # used to be __call__
        """Multiply by weight matrix and apply activation units"""

        g = torch.matmul(x, self.W)  # shape = (?, self.size)
        self.output = []

        in_i = 0  # input index
        out_i = 0  # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            self.output.append(self.funcs[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            self.output.append(self.funcs[out_i](g[:, in_i], g[:, in_i + 1]))
            in_i += 2
            out_i += 1

        self.output = torch.stack(self.output, dim=1)

        return self.output

    def get_weight(self):
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self):
        return self.W.clone()


class SymbolicLayerL0(SymbolicLayer):
    def __init__(self, in_dim=None, funcs=None, initial_weight=None, init_stddev=0.1,
                 bias=False, droprate_init=0.5, lamba=1.,
                 beta=2 / 3, gamma=-0.1, zeta=1.1, epsilon=1e-6):
        super().__init__(in_dim=in_dim, funcs=funcs, initial_weight=initial_weight, init_stddev=init_stddev)

        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.in_dim = in_dim
        self.eps = None

        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.epsilon = epsilon

        if self.use_bias:
            self.bias = nn.Parameter(0.1 * torch.ones((1, self.out_dim)))
        self.qz_log_alpha = nn.Parameter(torch.normal(mean=np.log(1 - self.droprate_init) - np.log(self.droprate_init),
                                                      std=1e-2, size=(in_dim, self.out_dim)))

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.qz_log_alpha) / self.beta)
        return y * (self.zeta - self.gamma) + self.gamma

    def sample_u(self, shape, reuse_u=False):
        """Uniform random numbers for concrete distribution"""
        if self.eps is None or not reuse_u:
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.eps = torch.rand(size=shape).to(DEVICE) * (1 - 2 * self.epsilon) + self.epsilon
        return self.eps

    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return torch.clamp(z, min=0, max=1)
        else:  # Mean of the hard concrete distribution
            pi = torch.sigmoid(self.qz_log_alpha)
            return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = torch.sigmoid(self.qz_log_alpha)
        return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = torch.clamp(z, min=0.0, max=1.0)
        return mask * self.W

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.W * self.get_z_mean()

    def loss(self):
        """Regularization loss term"""
        return torch.sum(torch.sigmoid(self.qz_log_alpha - self.beta * np.log(-self.gamma / self.zeta)))

    def forward(self, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        if sample:
            h = torch.matmul(x, self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        if self.use_bias:
            h = h + self.bias

        # shape of h = (?, self.n_funcs)

        output = []
        # apply a different activation unit to each column of h
        in_i = 0  # input index
        out_i = 0  # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            output.append(self.funcs[out_i](h[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](h[:, in_i], h[:, in_i + 1]))
            in_i += 2
            out_i += 1
        output = torch.stack(output, dim=1)
        return output


# A helper class that keeps track of meta updates
# It's done by replacing parameters with variables and applying updates to
# them.


class MetaModel:

    def __init__(self, model):
        self.model = model

    def reset(self):
        def reset_params(model):
            for module in model.children():
                if module._parameters and module._parameters['weight'] is not None:
                    module._parameters['weight'] = Variable(module._parameters['weight'].data)
                if module._parameters and module._parameters['bias'] is not None:
                    module._parameters['bias'] = Variable(module._parameters['bias'].data)
                if type(module) == nn.Sequential or type(module) == BasicBlock:
                    reset_params(module)

        reset_params(self.model)

    def get_params(self):
        params = []
        for i_group, (name, p) in enumerate(self.model.named_parameters()):
            # cur_sz = int(np.prod(p.size()))
            # params.append(detach_var(p.view(cur_sz, 1)))
            params.append(p.view(-1))

        return params

    def get_flat_params(self):
        params = []

        def get_parameter(model):
            for module in model.children():
                if module._parameters and module._parameters['weight'] is not None:
                    params.append(module._parameters['weight'].view(-1))
                if module._parameters and module._parameters['bias'] is not None:
                    params.append(module._parameters['bias'].view(-1))
                if type(module) == nn.Sequential or type(module) == BasicBlock:
                    get_parameter(module)

        get_parameter(self.model)
        '''
        for module in self.model.children():
            if module._parameters and module._parameters['weight'] is not None:
                params.append(module._parameters['weight'].view(-1))
            if module._parameters and module._parameters['bias'] is not None:
                params.append(module._parameters['bias'].view(-1))
        '''
        return torch.cat(params)

    def set_params(self, flat_params):
        offset = 0
        for i_group, (name, p) in enumerate(self.model.named_parameters()):
            shape = p.shape
            length = p.view(-1).size()[0]
            p_i = flat_params[offset: offset + length].view(*shape)
            self.model.state_dict()[name][:] = p_i
            offset += length

    def set_flat_params(self, flat_params):
        # Restore original shapes
        offset = 0

        def set_parameters(model, offset):
            for module in model.children():
                if module._parameters and module._parameters['weight'] is not None:
                    weight_shape = module._parameters['weight'].size()
                    weight_flat_size = reduce(mul, weight_shape, 1)
                    module._parameters['weight'] = flat_params[offset:offset + weight_flat_size].view(*weight_shape)
                    offset += weight_flat_size
                if module._parameters and module._parameters['bias'] is not None:
                    bias_shape = module._parameters['bias'].size()
                    bias_flat_size = reduce(mul, bias_shape, 1)
                    module._parameters['bias'] = flat_params[offset:offset + bias_flat_size].view(*bias_shape)
                    offset += bias_flat_size
                if type(module) == nn.Sequential or type(module) == BasicBlock:
                    set_parameters(module, offset)

        set_parameters(self.model, offset)
        '''
        for i, module in enumerate(self.model.children()):
            if module._parameters and module._parameters['weight'] is not None:
                weight_shape = module._parameters['weight'].size()
            if module._parameters and module._parameters['bias'] is not None:
                bias_shape = module._parameters['bias'].size()

            if module._parameters and module._parameters['weight'] is not None:
                weight_flat_size = reduce(mul, weight_shape, 1)
            if module._parameters and module._parameters['bias'] is not None:
                bias_flat_size = reduce(mul, bias_shape, 1)

            if module._parameters and module._parameters['weight'] is not None:
                module._parameters['weight'] = flat_params[
                                               offset:offset + weight_flat_size].view(*weight_shape)

            if module._parameters and module._parameters['bias'] is not None:
                module._parameters['bias'] = flat_params[
                                             offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                    *bias_shape)

            if module._parameters and module._parameters['weight'] is not None:
                offset += weight_flat_size
            if module._parameters and module._parameters['bias'] is not None:
                offset += bias_flat_size
            # offset += weight_flat_size + bias_flat_size
        '''

    def copy_params_from(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelA.data.copy_(modelB.data)

    def copy_params_to(self, model):
        for modelA, modelB in zip(self.model.parameters(), model.parameters()):
            modelB.data.copy_(modelA.data)
