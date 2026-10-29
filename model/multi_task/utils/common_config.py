import os
import importlib
import sys
import types
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.custom_collate import collate_mil
import pdb


def get_backbone(p):
    """ Return the backbone """
    if p['backbone'].startswith('swin_'):
        # MTMamba only uses mmseg's pure-PyTorch Swin backbone.  The lightweight
        # mmcv package does not ship ``mmcv._ext``; importing mmseg nevertheless
        # eagerly registers unrelated detection/segmentation CUDA operators.
        # Provide explicit never-called placeholders so Swin can be imported
        # without compiling the entire mmcv-full operator suite.
        try:
            importlib.import_module('mmcv._ext')
        except ModuleNotFoundError:
            unavailable_ext = types.ModuleType('mmcv._ext')
            unavailable_ext.__file__ = '<unused-mtmamba-mmcv-ext>'

            def _unavailable_mmcv_op(name):
                def unavailable(*args, **kwargs):
                    del args, kwargs
                    raise RuntimeError(
                        f'mmcv CUDA operator {name!r} is unavailable; '
                        'the static MTMamba Swin path must not call it'
                    )
                return unavailable

            unavailable_ext.__getattr__ = _unavailable_mmcv_op
            sys.modules['mmcv._ext'] = unavailable_ext

        from mmseg.models.backbones.swin import SwinTransformer

        backbone_specs = {
            'swin_tiny': {
                'window_size': 7,
                'embed_dims': 96,
                'depths': (2, 2, 6, 2),
                'num_heads': (3, 6, 12, 24),
                'pretrained_filename': 'mmseg_swin_tiny_patch4_window7_224.pth',
            },
            'swin_small': {
                'window_size': 7,
                'embed_dims': 96,
                'depths': (2, 2, 18, 2),
                'num_heads': (3, 6, 12, 24),
                'pretrained_filename': 'mmseg_swin_small_patch4_window7_224.pth',
            },
            'swin_base': {
                'window_size': 7,
                'embed_dims': 128,
                'depths': (2, 2, 18, 2),
                'num_heads': (4, 8, 16, 32),
                'pretrained_filename': 'mmseg_swin_base_patch4_window7_224.pth',
            },
            'swin_large': {
                'window_size': 12,
                'embed_dims': 192,
                'depths': (2, 2, 18, 2),
                'num_heads': (6, 12, 24, 48),
                'pretrained_filename': 'mmseg_swin_large_patch4_window12_384_22kto1k.pth',
            },
        }
        if p['backbone'] not in backbone_specs:
            raise NotImplementedError

        spec = backbone_specs[p['backbone']]
        backbone_channels = None
        ppath = p.get('pretrained_backbone_dir', './pretrained_ckpts/')
        load_pretrained = bool(p.get('load_pretrained', True))
        pretrained_filename = p.get('pretrained_backbone_filename', spec['pretrained_filename'])
        pretrained_path = f'{ppath}/{pretrained_filename}' if load_pretrained else None
        backbone = SwinTransformer(
            patch_size=4,
            window_size=spec['window_size'],
            embed_dims=spec['embed_dims'],
            depths=spec['depths'],
            num_heads=spec['num_heads'],
            pretrain_img_size=p.TRAIN.SCALE,
            pretrained=pretrained_path,
        )
        backbone.init_weights()
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_model(p):
    if p.get('run_mode') == 'infer':
        from copy import deepcopy
        p = deepcopy(p)
        p.load_pretrained = False
        p.init_model_path = ''
        p.resume_flow_net = ''
        p.event_refine_checkpoint = ''
    """ Return the model """

    if p['model'] == 'MTMamba':
        backbone, backbone_channels = get_backbone(p)
        from models.MTMamba import MTMamba
        model = MTMamba(p, backbone)
    elif p['model'] == 'MTMamba_plus':
        backbone, backbone_channels = get_backbone(p)
        from models.MTMamba_plus import MTMamba_plus
        model = MTMamba_plus(p, backbone)
    elif p['model'] in {'LiFRV1', 'LiFRV2', 'MTMamba_plus_anytime'}:
        backbone, backbone_channels = get_backbone(p)
        if p['model'] == 'LiFRV1':
            from model.multi_task.v1 import Model
        elif p['model'] == 'LiFRV2':
            from model.multi_task.v2 import Model
        else:
            from models.MTMamba_plus_anytime import MTMamba_plus_anytime as Model
        model = Model(p, backbone)
    else:
        raise NotImplementedError('Unknown model {}'.format(p['model']))

    init_model_path = p.get('init_model_path', '')
    if init_model_path:
        if not os.path.isfile(init_model_path):
            raise FileNotFoundError(f'init_model_path does not exist: {init_model_path}')
        checkpoint = torch.load(init_model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            for key in ('model', 'state_dict', 'model_state_dict'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError(f'Unsupported checkpoint format for {init_model_path}: {type(checkpoint)}')

        cleaned = {}
        for key, value in checkpoint.items():
            key = key.replace('module.', '', 1) if key.startswith('module.') else key
            cleaned[key] = value

        from model.modules.memory_utils import remap_memory_state
        cleaned = remap_memory_state(cleaned)
        model_state = model.state_dict()
        compatible = {
            key: value for key, value in cleaned.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        stage = p.get('training_stage', 'backbone')
        required = tuple(name for name in model_state if not name.startswith('event_refine.'))
        if stage == 'propagation':
            required = tuple(name for name in required if not name.startswith(('flow_net.', 'softsplat_net.', 'hrm.', 'fusion_attens.')))
        if stage in {'propagation', 'completion'}:
            missing = set(required) - compatible.keys()
            if missing:
                raise RuntimeError(f'Incomplete {stage} initializer: {sorted(missing)[:20]}')
        msg = model.load_state_dict(compatible, strict=False)
        print(f"[init_model] loaded {len(compatible)} compatible keys from {init_model_path}")
        print(msg)

    stage = p.get('training_stage', 'backbone')
    if stage == 'completion':
        trainable = ('event_refine.fusions.', 'event_refine.gates.', 'expand_layers.',
                     'concat_layers.', 'block_stm.', 'block_ctm.', 'final_expand.', 'final_project.')
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(trainable))
        model._frozen_eval_prefixes = ('backbone', 'flow_net', 'softsplat_net', 'hrm',
                                       'event_refine.event_adapter')
    elif stage == 'propagation':
        model.backbone.requires_grad_(False)
        model._frozen_eval_prefixes = ('backbone',)
    elif stage != 'backbone':
        raise ValueError(f'Unknown training_stage: {stage}')
    return model


"""
    Transformations, datasets and dataloaders
"""
def get_transformations(p):
    """ Return transformations for training and evaluationg """
    from data import transforms
    import torchvision

    # Training transformations
    if p['train_db_name'] in ['NYUD', 'PASCALContext', 'DSEC', 'M3ED', 'shf_emergy']:
        train_transforms = torchvision.transforms.Compose([ # from ATRC
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=p.TRAIN.SCALE, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TRAIN.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])

        # Testing 
        valid_transforms = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=p.TEST.SCALE),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
        return train_transforms, valid_transforms

    else:
        return None, None


def _uses_temporal_loader(p):
    return p.get('dataset_variant', 'standard') in {'anytime', 'query', 'refine', 'temporal'}


def _dataset_tasks(p, split):
    return set(p.get('eval_tasks', p.TASKS.NAMES) if split == 'val' else p.TASKS.NAMES)


def _build_dsec_dataset(
    p,
    split,
    transforms,
    tasks=None,
    time_window_steps=None,
):
    tasks = set(tasks) if tasks is not None else _dataset_tasks(p, split)
    common = dict(
        root=p.db_paths['DSEC'],
        split=split,
        transform=transforms,
        retname=True,
        do_semseg='semseg' in tasks,
        do_depth='depth' in tasks,
        train_sequences=getattr(p, 'train_sequences', None),
        val_sequences=getattr(p, 'val_sequences', None),
    )
    if _uses_temporal_loader(p):
        from data.dsec_anytime import DSECAnytime
        return DSECAnytime(
            **common,
            time_window_steps=int(
                getattr(p, 'time_window_steps', 1)
                if time_window_steps is None
                else time_window_steps
            ),
            event_bins=int(getattr(p, 'event_bins', 20)),
        )
    from data.dsec import DSEC
    return DSEC(**common)


def _build_m3ed_dataset(p, split, transforms):
    tasks = _dataset_tasks(p, split)
    common = dict(
        root=p.db_paths['M3ED'],
        semantic_root=getattr(p, 'm3ed_semantic_root', None),
        split=split,
        transform=transforms,
        retname=True,
        do_semseg='semseg' in tasks,
        do_depth='depth' in tasks,
    )
    if _uses_temporal_loader(p):
        from data.m3ed_anytime import M3EDAnytime
        return M3EDAnytime(
            **common,
            time_window_steps=int(getattr(p, 'time_window_steps', 1)),
            event_bins=int(getattr(p, 'event_bins', 20)),
        )
    from data.m3ed import M3ED
    return M3ED(**common)


def _build_shf_emergy_dataset(p, split, transforms):
    tasks = _dataset_tasks(p, split)
    common = dict(
        root=p.db_paths['shf_emergy'],
        split=split,
        transform=transforms,
        retname=True,
        do_semseg='semseg' in tasks,
        do_depth='depth' in tasks,
    )
    if _uses_temporal_loader(p):
        from data.shf_emergy_anytime import SHFEmergyAnytime
        return SHFEmergyAnytime(
            **common,
            time_window_steps=int(getattr(p, 'time_window_steps', 1)),
            event_bins=int(getattr(p, 'event_bins', 20)),
        )
    from data.shf_emergy import SHFEmergy
    return SHFEmergy(**common)



def get_train_dataset(p, transforms=None):
    """ Return the train dataset """

    db_name = p['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['train'], transform=transforms, retname=True,
                                          do_semseg='semseg' in p.TASKS.NAMES,
                                          do_edge='edge' in p.TASKS.NAMES,
                                          do_normals='normals' in p.TASKS.NAMES,
                                          do_sal='sal' in p.TASKS.NAMES,
                                          do_human_parts='human_parts' in p.TASKS.NAMES,
                                          overfit=False)

    if db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='train', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES, overfit=False)

    if db_name == 'Cityscapes':
        from data.cityscapes import CITYSCAPES
        database = CITYSCAPES(p, p.db_paths['Cityscapes'], split=["train"], is_transform=True,
                    img_size=p.TRAIN.SCALE, augmentations=None, 
                    task_list=p.TASKS.NAMES)

    if db_name == 'DSEC':
        database = _build_dsec_dataset(p, 'train', transforms)

    if db_name == 'M3ED':
        database = _build_m3ed_dataset(p, 'train', transforms)

    if db_name == 'shf_emergy':
        database = _build_shf_emergy_dataset(p, 'train', transforms)

    return database


def get_train_dataloader(p, dataset, sampler):
    """ Return the train dataloader """
    collate = collate_mil
    trainloader = DataLoader(dataset, batch_size=p['trBatch'], drop_last=True,
                             num_workers=p['nworkers'], collate_fn=collate, pin_memory=True, sampler=sampler)
    return trainloader


def get_test_dataset(p, transforms=None):
    """ Return the test dataset """

    db_name = p['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'PASCALContext':
        from data.pascal_context import PASCALContext
        database = PASCALContext(p.db_paths['PASCALContext'], download=False, split=['val'], transform=transforms, retname=True,
                                      do_semseg='semseg' in p.TASKS.NAMES,
                                      do_edge='edge' in p.TASKS.NAMES,
                                      do_normals='normals' in p.TASKS.NAMES,
                                      do_sal='sal' in p.TASKS.NAMES,
                                      do_human_parts='human_parts' in p.TASKS.NAMES,
                                    overfit=False)
    
    elif db_name == 'NYUD':
        from data.nyud import NYUD_MT
        database = NYUD_MT(p.db_paths['NYUD_MT'], download=False, split='val', transform=transforms, do_edge='edge' in p.TASKS.NAMES, 
                                    do_semseg='semseg' in p.TASKS.NAMES, 
                                    do_normals='normals' in p.TASKS.NAMES, 
                                    do_depth='depth' in p.TASKS.NAMES)

    elif db_name == 'Cityscapes':
        from data.cityscapes import CITYSCAPES
        database = CITYSCAPES(p, p.db_paths['Cityscapes'], split=["val"], is_transform=True,
                    img_size=p.TEST.SCALE, augmentations=None, 
                    task_list=p.TASKS.NAMES)
    
    elif db_name == 'DSEC':
        database = _build_dsec_dataset(p, 'val', transforms)

    elif db_name == 'M3ED':
        database = _build_m3ed_dataset(p, 'val', transforms)

    elif db_name == 'shf_emergy':
        database = _build_shf_emergy_dataset(p, 'val', transforms)

    else:
        raise NotImplementedError(f'Unsupported val_db_name: {db_name}')

    return database


def get_test_dataloader(p, dataset):
    """ Return the validation dataloader """
    collate = collate_mil
    testloader = DataLoader(dataset, batch_size=p['valBatch'], shuffle=False, drop_last=False,
                            num_workers=p['nworkers'], pin_memory=True, collate_fn=collate)
    return testloader


def get_test_dataloaders(p, transforms=None):
    """Build validation loaders, optionally one independent loader per task.

    DSEC semantic and depth validation annotations belong to disjoint sequences.
    A task-specific loader prevents either modality from being represented by a
    synthetic ignore/zero target while preserving the common evaluation API.
    """
    tasks = list(p.get('eval_tasks', p.TASKS.NAMES))
    if (
        p['val_db_name'] == 'DSEC'
        and p.get('separate_eval_tasks', False)
        and len(tasks) > 1
    ):
        task_steps = p.get('eval_time_window_steps', {})
        return {
            task: get_test_dataloader(
                p,
                _build_dsec_dataset(
                    p,
                    'val',
                    transforms,
                    tasks=(task,),
                    time_window_steps=task_steps.get(
                        task,
                        getattr(p, 'time_window_steps', 1),
                    ),
                ),
            )
            for task in tasks
        }

    return get_test_dataloader(p, get_test_dataset(p, transforms))


""" 
    Loss functions 
"""
def get_loss(p, task=None):
    """ Return loss function for a specific task """

    if task == 'edge':
        from losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    elif task == 'semseg' or task == 'human_parts':
        semseg_loss_cfg = p['loss_kwargs'].get('semseg_loss', {}) if 'loss_kwargs' in p else {}
        semseg_loss_name = semseg_loss_cfg.get('name', 'ce').lower() if task == 'semseg' else 'ce'

        if task == 'semseg' and semseg_loss_name in {'ohem', 'ohem_ce', 'ohem_cross_entropy'}:
            from losses.loss_functions import OhemCrossEntropy
            criterion = OhemCrossEntropy(
                ignore_index=p.ignore_index,
                thresh=semseg_loss_cfg.get('thresh', 0.7),
                aux_weights=tuple(semseg_loss_cfg.get('aux_weights', (1.0, 0.0))),
            )
        else:
            from losses.loss_functions import CrossEntropyLoss
            criterion = CrossEntropyLoss(ignore_index=p.ignore_index)

    elif task == 'normals':
        from losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)

    elif task == 'sal':
        from losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 

    elif task == 'depth':
        from losses.loss_functions import L1Loss
        depth_cfg = p['loss_kwargs'].get('depth_loss', {})
        criterion = L1Loss(
            ignore_index=p.ignore_index,
            min_value=depth_cfg.get('min_depth'),
            max_value=depth_cfg.get('max_depth'),
            value_weight_ranges=depth_cfg.get('weight_ranges', ()),
        )

    else:
        criterion = None

    return criterion


def get_criterion(p):
    if p['loss_kwargs']['loss_scheme'] == 'log':
        from losses.loss_schemes import MultiTaskLoss_log
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MultiTaskLoss_log(p, p.TASKS.NAMES, loss_ft, loss_weights)
    else:
        from losses.loss_schemes import MultiTaskLoss
        loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
        loss_weights = p['loss_kwargs']['loss_weights']
        return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)


"""
    Optimizers and schedulers
"""
def get_optimizer(p, model, iters_per_epoch=None):
    """ Return optimizer for a given model and setup """

    print('Optimizer uses a single parameter group - (Default)')
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise ValueError('The optimizer received no trainable parameters')

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
                                params,
                                lr=p['optimizer_kwargs']['lr'],
                                betas=(0.9, 0.999), eps=1e-08,
                                weight_decay=p['optimizer_kwargs']['weight_decay'], amsgrad=False)
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    # get scheduler
    scheduler_name = str(p.scheduler).lower()
    scheduler_kwargs = p.get('scheduler_kwargs', {})
    if scheduler_name == 'poly':
        from utils.train_utils import PolynomialLR
        scheduler = PolynomialLR(
            optimizer,
            p.max_iter,
            gamma=scheduler_kwargs.get('power', 0.9),
            min_lr=scheduler_kwargs.get('min_lr', 0),
        )
    elif scheduler_name == 'warmuppolylr':
        from utils.train_utils import WarmupPolynomialLR
        if 'warmup_iter' in scheduler_kwargs:
            warmup_iter = int(scheduler_kwargs['warmup_iter'])
        elif 'warmup_epochs' in scheduler_kwargs and iters_per_epoch is not None:
            warmup_iter = int(float(scheduler_kwargs['warmup_epochs']) * int(iters_per_epoch))
        else:
            warmup_iter = 0
        scheduler = WarmupPolynomialLR(
            optimizer,
            p.max_iter,
            gamma=scheduler_kwargs.get('power', 0.95),
            warmup_iterations=warmup_iter,
            warmup_ratio=scheduler_kwargs.get('warmup_ratio', 0.1),
            min_lr=scheduler_kwargs.get('min_lr', 0),
        )
        print(
            f"Using warmuppolylr scheduler: max_iter={p.max_iter} "
            f"warmup_iter={warmup_iter} power={scheduler_kwargs.get('power', 0.95)} "
            f"warmup_ratio={scheduler_kwargs.get('warmup_ratio', 0.1)}"
        )
    else:
        scheduler = None

    return scheduler, optimizer
