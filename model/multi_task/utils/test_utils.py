# By Hanrong Ye
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import autocast_forward, float_tensors, get_output, mkdir_if_missing, to_cuda
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os
from collections.abc import Mapping

@torch.no_grad()
def test_phase(
    p,
    test_loader,
    model,
    epoch,
    save_edge=False,
    tasks=None,
):
    if isinstance(test_loader, Mapping):
        results = {}
        for task, task_loader in test_loader.items():
            task_results = test_phase(
                p,
                task_loader,
                model,
                epoch,
                save_edge=save_edge,
                tasks=(task,),
            )
            results.update(task_results)
        return results

    tasks = list(tasks or p.get('eval_tasks', p.TASKS.NAMES))

    performance_meter = PerformanceMeter(p, tasks)

    model.eval()

    if save_edge:
        tasks_to_save = ['edge'] if 'edge' in tasks else []
        save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
        for save_dir in save_dirs.values():
            mkdir_if_missing(save_dir)
    eval_model = model.module if hasattr(model, 'module') else model
    
    for cpu_batch in tqdm(test_loader):
        batch = to_cuda(cpu_batch)
        model_input = batch if p.get('batch_input', False) else batch['image']
        targets = {task: batch[task] for task in tasks}
        with autocast_forward(p):
            output = eval_model(model_input)
        output = float_tensors(output)
        performance_meter.update(
            {task: get_output(output[task], task) for task in tasks}, targets,
        )
        if save_edge:
            for task in tasks_to_save:
                save_model_pred_for_one_task(p, batch, output, save_dirs, task, epoch=epoch)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
