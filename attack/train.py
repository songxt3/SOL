import torchattacks
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from meta_optimizer import AttackModel, ExprAttackModel, Expr
import functions
import torch
import torch.optim as optim
import pretty_print
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from tqdm import tqdm

torch.manual_seed(428)

device = "cuda:1"

images, labels = load_cifar10(n_examples=10000)
images = images.to(device)
labels = labels.to(device)
dataset = TensorDataset(images, labels)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataiter = cycle(dataloader)


model_name = "Carmon2019Unlabeled"

model = load_model(model_name, norm='Linf').to(device)

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
meta_optimizer = AttackModel(n_layers=1, in_dim=14, funcs=funcs, initial_weights=[
    # kind of a hack for truncated normal
    torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
    torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
    torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
    torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
])

meta_optimizer.to(device)

optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-4, weight_decay=1e-5, amsgrad=True)

images = images.clone().detach().to(device)
labels = labels.clone().detach().to(device)

loss_func = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
apgd_atk = torchattacks.APGD(model, steps=150)

for epoch in tqdm(range(250), desc='Training:'):
    images, labels = next(dataiter)
    adv_images = apgd_atk(images, labels)
    steps = 100
    begin_images = images
    begin_images.requires_grad = True
    for i in range(steps):
        outputs = model(begin_images)
        cost = loss_func(outputs, labels)
        grad = torch.autograd.grad(cost, begin_images, retain_graph=False, create_graph=False)
        update = meta_optimizer.meta_update(grad[0], i+1, 100)
        begin_images = begin_images + 2 / 255 * update
        # delta = torch.clamp(begin_images - images, min=-8 / 255, max=8 / 255)
        # begin_images = torch.clamp(images + delta, min=0, max=1)

    loss = criterion(begin_images, adv_images)
    meta_optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    weights = meta_optimizer.get_weights()
    expr = pretty_print.network(weights, funcs,
                                ["m", "v", "g", "g2", "g3", "ag", "sg", "sm", "ad", "rs", "ld", "cd", "1",
                                 "2"])

expr_optimizer = ExprAttackModel(expr)

model_list = ['Carmon2019Unlabeled', 'Gowal2020Uncovering_28_10_extra', 'Sehwag2020Hydra', 'Wu2020Adversarial', 'Gowal2020Uncovering_34_20', 'Engstrom2019Robustness', 'Wong2020Fast']
images, labels = load_cifar10(n_examples=256)
for model_name in model_list:
    model = load_model(model_name, norm='Linf').to(device)
    standard_acc = clean_accuracy(model, images.to(device), labels.to(device))

    optimizer = ExprAttackModel(expr)
    expr_atk = Expr(model=model, device=device, eps=8/255, alpha=2/255, steps=100, meta_optimizer=optimizer)
    adv_images_3 = expr_atk(images, labels)
    expr_acc = clean_accuracy(model, adv_images_3, labels)

    pgd_atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=100, random_start=False)
    adv_images = pgd_atk(images, labels)
    pgd_acc = clean_accuracy(model, adv_images, labels)

    print('Model: {}'.format(model_name))
    print('- Standard Acc: {}'.format(1-standard_acc))
    print('- PGD Acc: {}'.format(1-pgd_acc))
    print('- Expr Accuracy: {}'.format(1-expr_acc))
