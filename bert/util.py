from transformers.optimization import Adafactor, AdamW, get_scheduler
from typing import Any, Dict, Union
import math
import torch
from datasets import load_metric


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def get_metrics(task: str):
    task_to_metric = {
        "cola": ["matthews_correlation", None],
        "sst2": ["accuracy", None],
        "mrpc": ["f1", "accuracy"],
        "stsb": ["pearsonr", None],
        "qqp": ["f1", "accuracy"],
        "mnli": ["accuracy", None],
        "mnli-mm": ["accuracy", None],
        "qnli": ["accuracy", None],
        "rte": ["accuracy", None],
        "wnli": ["accuracy", None],
    }
    metric = load_metric(task_to_metric[task][0])
    metric_1 = load_metric(task_to_metric[task][1]) if task_to_metric[task][1] else None
    return metric, metric_1


def compute_metrics(predictions, references, metric):
    if f"{metric.__class__.__name__}" != 'Pearsonr':
        predictions = torch.argmax(predictions, dim=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=references)


def prepare_inputs(inputs: Dict[str, Union[torch.Tensor, Any]], device) -> Dict[str, Union[torch.Tensor, Any]]:
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
    return inputs


def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


def create_optimizer(model, adafactor=None, weight_decay=0.01, learning_rate=2e-5,
                     adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8):
    """
    Setup the optimizer.

    We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
    """
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer_cls = Adafactor if adafactor else AdamW
    if adafactor:
        optimizer_cls = Adafactor
        optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
    else:
        optimizer_cls = AdamW
        optimizer_kwargs = {
            "betas": (adam_beta1, adam_beta2),
            "eps": adam_epsilon,
        }
    optimizer_kwargs["lr"] = learning_rate
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer


def create_scheduler(optimizer, lr_scheduler_type: str = "linear", num_training_steps: int = 10,
                     warmup_steps: int = 0, warmup_ratio: float = 0.0):
    """
    Setup the scheduler. The optimizer of the trainer must have been set up before this method is called.

    Args:
        num_training_steps (int): The number of training steps to do.
    """
    warmup_steps = (
        warmup_steps
        if warmup_steps > 0
        else math.ceil(num_training_steps * warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler
