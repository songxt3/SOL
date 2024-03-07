import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
from torchattacks.attack import Attack
import functions

DEVICE = torch.device("cuda:1")


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


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


class AttackModel(nn.Module):
    def __init__(self, n_layers, in_dim=1, funcs=None, initial_weights=None, beta1=0.9, beta2=0.999):
        super(AttackModel, self).__init__()
        self.beta1 = torch.tensor(beta1).type(torch.float)
        self.beta2 = torch.tensor(beta2).type(torch.float)
        self.depth = n_layers
        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i],
                                      in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        self.hidden_layers = nn.Sequential(*layers)

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
        return fea_array

    def meta_update(self, grads, step, T):
        flat_grads = grads.view(-1).unsqueeze(-1)
        if step == 1:
            self.mt_all = torch.zeros(flat_grads.shape).to(DEVICE)
            self.vt_all = torch.zeros(flat_grads.shape).to(DEVICE)
        fea_array = self.build_fea(flat_grads, step, T)
        for i in range(len(fea_array)):
            if fea_array[i].size(0) != flat_grads.size(0):
                fea_array[i] = fea_array[i].expand_as(flat_grads)
        inputs = Variable(torch.cat(fea_array, 1))
        outputs = self(inputs)
        outputs = outputs.view(grads.shape)
        return outputs

    def forward(self, input, sample=True, reuse_u=False):
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)

        h = torch.matmul(h, self.output_weight)
        return h

    def get_weights(self):
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]


class ExprAttackModel(nn.Module):
    def __init__(self, expr, beta1=0.9, beta2=0.999) -> None:
        super(ExprAttackModel, self).__init__()
        self.expr = str(expr)
        self.expr = self.expr.replace('exp', 'torch.exp')
        self.expr = self.expr.replace('sin', 'torch.sin')
        print(expr)
        self.vt_all = None
        self.mt_all = None
        self.beta1 = torch.tensor(beta1).type(torch.float)
        self.beta2 = torch.tensor(beta2).type(torch.float)

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
        return fea_array

    def meta_update(self, grads, step, T):
        flat_gards = grads.view(-1).unsqueeze(-1)
        if step == 1:
            self.mt_all = torch.zeros(flat_gards.shape).to(DEVICE)
            self.vt_all = torch.zeros(flat_gards.shape).to(DEVICE)
        fea_array = self.build_fea(flat_gards, step, T)
        for i in range(len(fea_array)):
            if fea_array[i].size(0) != flat_gards.size(0):
                fea_array[i] = fea_array[i].expand_as(flat_gards)
        inputs = Variable(torch.cat(fea_array, 1))

        outputs = self(inputs)
        outputs = outputs.view(grads.shape)
        return outputs

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

        def sign(x):
            if torch.is_tensor(x):
                return torch.sign(x)
            else:
                return np.sign(x)

        updates = eval(self.expr)
        return updates.unsqueeze(-1)


class Expr(Attack):
    def __init__(self, model, device, eps, alpha, steps, meta_optimizer):
        super().__init__('Expr', model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.meta_optimizer = meta_optimizer

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # adv_images = self.meta_optimizer.meta_update(images, 1, 10)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)

            update = self.meta_optimizer.meta_update(grad[0], i + 1, self.steps)

            adv_images = adv_images.detach() + self.alpha * update
            delta = torch.clamp(adv_images - images,
                                min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
