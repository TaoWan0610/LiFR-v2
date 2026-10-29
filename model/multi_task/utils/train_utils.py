# Rewritten based on MTI-Net by Hanrong Ye
# Original authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)


import os, json
from evaluation.evaluate_utils import count_improvement
from utils.utils import autocast_forward, float_tensors, to_cuda
import torch
import torch.distributed as dist
from tqdm import tqdm
from utils.test_utils import test_phase


def _safe_result_stem(version_name: str) -> str:
    return version_name.replace('\\', '_').replace('/', '_')


def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)


def build_checkpoint_state(model, optimizer, scheduler, epoch, iter_count, best_imp):
    checkpoint_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'iter_count': iter_count,
        'best_imp': best_imp,
    }
    if scheduler is not None:
        checkpoint_state['scheduler'] = scheduler.state_dict()
    return checkpoint_state


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _barrier_if_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _keep_frozen_modules_in_eval(model):
    """Keep explicitly frozen stateful modules from updating running stats."""
    unwrapped = _unwrap_model(model)
    prefixes = tuple(getattr(unwrapped, '_frozen_eval_prefixes', ()))
    if not prefixes:
        return
    for name, module in unwrapped.named_modules():
        if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            module.eval()


def _validate_and_save(p, loader, model, optimizer, scheduler, epoch, iter_count, best_imp, *, final=False):
    """Use the same validation and checkpoint policy at epoch and iteration limits."""
    result = test_phase(p, loader, _unwrap_model(model), epoch)
    print(f'Validation at epoch {epoch}: {result}')
    if not final:
        name = f'{_safe_result_stem(p.version_name)}_{epoch}.txt'
        with open(os.path.join(p['save_dir'], name), 'w') as file:
            json.dump(result, file, indent=4)
    score = count_improvement(
        p['train_db_name'], result, list(p.get('eval_tasks', p['TASKS']['NAMES'])),
    )
    improved = score > best_imp
    if improved:
        best_imp = score
    state = build_checkpoint_state(model, optimizer, scheduler, epoch, iter_count, best_imp)
    torch.save(state, p['checkpoint'])
    metadata = {**result, 'epoch': epoch, 'best_imp': best_imp}
    names = (['best'] if improved else []) + (['last'] if final else [])
    for name in names:
        path = p['checkpoint'].replace('checkpoint.pth.tar', f'{name}_model.pth.tar')
        torch.save(state, path)
        path = p['checkpoint'].replace('checkpoint.pth.tar', f'{name}.txt')
        with open(path, 'w') as file:
            json.dump(metadata, file, indent=4)
    return best_imp


def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, 
                scheduler, epoch, tb_writer, iter_count, best_imp):
    """ Vanilla training with fixed loss weights """
    model.train()
    _keep_frozen_modules_in_eval(model)
    if p.get('training_stage') == 'completion':
        active = epoch >= int(p.get('gate_start_epoch', 10))
        _unwrap_model(model).event_refine.train_gate(active)

    primary = not dist.is_initialized() or dist.get_rank() == 0
    progress_bar = tqdm(
        train_loader,
        disable=not primary,
        dynamic_ncols=True,
        leave=True,
    )
    if primary:
        progress_bar.set_description(f'Epoch {epoch}')

    for cpu_batch in progress_bar:
        # Forward pass
        batch = to_cuda(cpu_batch)
        model_input = batch if p.get('batch_input', False) else batch['image']

        with autocast_forward(p):
            output = model(model_input)
        output = float_tensors(output)
        iter_count += 1
        # Measure loss
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)
        # get learning rate
        if scheduler is not None:
            lr = scheduler.get_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']
        loss_dict['lr'] = torch.tensor(lr)

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)

        if primary:
            progress_stats = {'iter': iter_count}
            for k, v in loss_dict.items():
                if hasattr(v, 'item'):
                    progress_stats[k] = f'{v.item():.4f}'
                else:
                    progress_stats[k] = f'{float(v):.4f}'
            progress_bar.set_postfix(progress_stats)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), **p.grad_clip_param)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        if iter_count >= p.max_iter:
            _barrier_if_distributed()
            if primary:
                best_imp = _validate_and_save(
                    p, test_dataloader, model, optimizer, scheduler,
                    epoch, iter_count, best_imp, final=True,
                )
            _barrier_if_distributed()
            return True, iter_count, best_imp

    if epoch >= 1:
        _barrier_if_distributed()
        if primary:
            best_imp = _validate_and_save(
                p, test_dataloader, model, optimizer, scheduler,
                epoch, iter_count, best_imp,
            )
            model.train()
        _barrier_if_distributed()
    return False, iter_count, best_imp


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


class WarmupPolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iterations,
        gamma=0.95,
        warmup_iterations=0,
        warmup_ratio=0.1,
        min_lr=0.,
        last_epoch=-1,
    ):
        self.max_iterations = int(max_iterations)
        self.gamma = gamma
        self.warmup_iterations = int(warmup_iterations)
        self.warmup_ratio = float(warmup_ratio)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        if self.warmup_iterations > 0 and step < self.warmup_iterations:
            alpha = step / float(max(self.warmup_iterations, 1))
            factor = self.warmup_ratio + (1.0 - self.warmup_ratio) * alpha
        else:
            denom = max(self.max_iterations - self.warmup_iterations, 1)
            progress = (step - self.warmup_iterations) / float(denom)
            progress = min(max(progress, 0.0), 1.0)
            factor = (1.0 - progress) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)
