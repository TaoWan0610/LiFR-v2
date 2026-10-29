from torch import nn
from torch.optim import AdamW, SGD


def get_optimizer(
    model: nn.Module,
    optimizer: str,
    lr: float,
    weight_decay: float = 0.01,
    include_frozen: bool = False,
):
    wd_params, nwd_params = [], []
    for p in model.parameters():
        if include_frozen or p.requires_grad:
            if p.dim() == 1:
                nwd_params.append(p)
            else:
                wd_params.append(p)
    
    params = [
        {"params": wd_params},
        {"params": nwd_params, "weight_decay": 0}
    ]

    if optimizer == 'adamw':
        return AdamW(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)


def get_backbone_optimizer(
    model: nn.Module,
    optimizer: str,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float = 0.01,
):
    """Set separate encoder and decoder learning rates for same-frame backbone training."""
    encoder = getattr(model, 'backbone', None)
    decoder = getattr(model, 'decode_head', None)
    if encoder is None or decoder is None:
        raise ValueError('Backbone optimizer requires model.backbone and model.decode_head')

    params = []
    for group_name, module, group_lr in (
        ('encoder', encoder, encoder_lr),
        ('decoder', decoder, decoder_lr),
    ):
        wd_params, nwd_params = [], []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            (nwd_params if parameter.dim() == 1 else wd_params).append(parameter)
        if wd_params:
            params.append({
                'params': wd_params,
                'lr': group_lr,
                'weight_decay': weight_decay,
                'group_name': group_name,
            })
        if nwd_params:
            params.append({
                'params': nwd_params,
                'lr': group_lr,
                'weight_decay': 0.0,
                'group_name': group_name,
            })

    if optimizer == 'adamw':
        return AdamW(params, betas=(0.9, 0.999), eps=1e-8)
    return SGD(params, momentum=0.9)


def get_mask_refine_optimizer(
    module: nn.Module,
    optimizer: str,
    refine_lr: float,
    event_encoder_lr: float,
    decoder_lr: float | None = None,
    gate_lr: float | None = None,
    base_lr: float | None = None,
    flow_lr: float | None = None,
    weight_decay: float = 0.01,
    event_encoder: nn.Module | None = None,
):
    event_branch = event_encoder or getattr(module, 'event_encoder', None)
    pretrained_b0 = getattr(event_branch, 'event_encoder', None)
    refine_stage_parameters = getattr(event_branch, 'refine_stage_parameters', None)
    event_parameters = (
        list(refine_stage_parameters())
        if callable(refine_stage_parameters)
        else list(event_branch.parameters()) if event_branch is not None else []
    )
    event_parameters = [
        parameter for parameter in event_parameters if parameter.requires_grad
    ]
    event_parameter_ids = {id(parameter) for parameter in event_parameters}
    refine_parameters, gate_parameters = [], []
    decoder_parameters, base_parameters, flow_parameters = [], [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) in event_parameter_ids:
            continue
        if decoder_lr is not None and name.startswith('decode_head.'):
            decoder_parameters.append(parameter)
        elif gate_lr is not None and (
            name.startswith('gates.')
            or name.startswith('mask_refiner.gates.')
        ):
            gate_parameters.append(parameter)
        elif flow_lr is not None and name.startswith('flow_net.'):
            flow_parameters.append(parameter)
        elif base_lr is not None and not name.startswith('mask_refiner.'):
            base_parameters.append(parameter)
        else:
            refine_parameters.append(parameter)

    module_groups = [
        ('refine', refine_parameters, refine_lr),
        ('gate', gate_parameters, gate_lr),
        ('event_encoder', event_parameters, event_encoder_lr),
    ]
    if decoder_parameters:
        module_groups.append(('decoder', decoder_parameters, decoder_lr))
    if base_parameters:
        module_groups.append(('base', base_parameters, base_lr))
    if flow_parameters:
        module_groups.append(('flow', flow_parameters, flow_lr))
    params = []
    for group_name, parameters, group_lr in module_groups:
        if not parameters:
            continue
        wd_params, nwd_params = [], []
        for parameter in parameters:
            (nwd_params if parameter.dim() == 1 else wd_params).append(parameter)
        params.extend([
            {'params': wd_params, 'lr': group_lr, 'weight_decay': weight_decay, 'group_name': group_name},
            {'params': nwd_params, 'lr': group_lr, 'weight_decay': 0.0, 'group_name': group_name},
        ])

    if optimizer == 'adamw':
        return AdamW(params, betas=(0.9, 0.999), eps=1e-8)
    return SGD(params, momentum=0.9)
