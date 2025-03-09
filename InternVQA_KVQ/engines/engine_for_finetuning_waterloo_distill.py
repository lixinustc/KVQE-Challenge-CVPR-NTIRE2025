import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup

import torch.distributed as dist
from timm.utils import ModelEma
import utils
from scipy.stats import spearmanr, pearsonr



def train_class_batch(model, samples, target, criterion):
    x,s,s_tgt = model(samples)
    outputs = s[:, 0]
    print("loss1:",x.dtype,"loss2",criterion(outputs, target).dtype,"loss3",criterion(s_tgt[:, 0], target).dtype)
    print('outputtype:',outputs.dtype,"targettype:",target.dtype,"s_tgt[:, 0]type:",s_tgt[:, 0].dtype)
    loss = x + criterion(outputs, target) + criterion(s_tgt[:, 0], target)
    print('outputs:',outputs,'target:',target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    try:
        return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
    except Exception:
        return 0


def train_one_epoch(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
        start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None,
        bf16=False,
    ):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "lr_scale" in param_group:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).bfloat16()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if loss_scaler is None:
            if not no_amp:
                samples = samples.bfloat16() if bf16 else samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with amp_autocast:
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)
        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()

        if loss_list_isnan or loss_list_isinf:
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            if loss_scaler != 'none':
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            else:
                loss /= update_freq
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    if max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = 0

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, ds=False, bf16=False):

    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).bfloat16()

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos)[:, 0]
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos)[:, 0]
                loss = criterion(output, target)

        metric_logger.update(loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
# def final_test(data_loader, model, device, file, amp_autocast, ds=True, no_amp=False, bf16=False):
def final_test(data_loader, model, device, file, ds=False, bf16=False):

    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).bfloat16()

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            output = model(videos)[:, 0]
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                output = model(videos)[:, 0]
                loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].float().cpu().numpy()), \
                                                str(float(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        metric_logger.update(loss=loss.item())

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}\n".format(loss.item()))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('*loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip().split(' ')
            name = line[0]
            data = float(line[1])
            label = float(line[2])
            chunk_nb = line[3]
            split_nb = line[4]
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    mse_loss = [x[0] for x in ans]
    final_loss = np.mean(mse_loss)
    return final_loss

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
def train_one_epoch_vqa(
        model: torch.nn.Module, criterion: torch.nn.Module,
        data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
        start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
        num_training_steps_per_epoch=None, update_freq=None,
        bf16=False,
    ):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "lr_scale" in param_group:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).bfloat16()
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        print('targets:',targets)
        if loss_scaler is None:
            samples = samples.bfloat16() if bf16 else samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        loss_list = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
        dist.all_gather(loss_list, loss)
        loss_list = torch.tensor(loss_list)
        loss_list_isnan = torch.isnan(loss_list).any()
        loss_list_isinf = torch.isinf(loss_list).any()

        if loss_list_isnan or loss_list_isinf:
            print(" ========== loss_isnan = {},  loss_isinf = {} ========== ".format(loss_list_isnan, loss_list_isinf))
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            if loss_scaler != 'none':
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]
            else:
                loss /= update_freq
                loss.backward()
                if (data_iter_step + 1) % update_freq == 0:
                    if max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = 0

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()

def validation_one_epoch_vqa(data_loader, model, device, ds=False, bf16=False):
    criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()
    results = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        result = dict()
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).bfloat16()

        # compute output
        if ds:
            videos = videos.bfloat16() if bf16 else videos.half()
            x,score,s_tgt = model(videos)
            output = score[:, 0]
            loss = x + criterion(output, target) + criterion(s_tgt[:,0], target)
            # loss = plcc_loss(output, target)+0.3*rank_loss(output,target)
        else:
            with torch.cuda.amp.autocast():
                x,score = model(videos)
                output = score[:, 0]
                loss = x + criterion(output, target) + criterion(s_tgt[:,0], target)
                # loss = plcc_loss(output, target)+0.3*rank_loss(output,target)
        result["pr_labels"] = output
        result["gt_label"] = target

        metric_logger.update(loss=loss.item())
        results.append(result)
    

    ## generate the demo video for video quality localization
    gt_labels = torch.cat([r["gt_label"].cpu() for r in results],dim=0).float().numpy()
    if bf16:
        pr_labels = torch.cat([r["pr_labels"].cpu().float() for r in results],dim=0).float().numpy()
    else:
        pr_labels = torch.cat([r["pr_labels"].cpu() for r in results],dim=0).float().numpy()
    # print('gt:',gt_labels)
    # print('pr:',pr_labels)
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]

    metric_logger.update(s=s.item())
    metric_logger.update(p=p.item())
    # k = kendallr(gt_labels, pr_labels)[0]
    # r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def gaussian(y, eps=1e-8):
    return (y - y.mean()) / (y.std() + 1e-8)


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def rescaled_l2_loss(y_pred, y):
    y_pred_rs = (y_pred - y_pred.mean()) / y_pred.std()
    y_rs = (y - y.mean()) / (y.std() + eps)
    return torch.nn.functional.mse_loss(y_pred_rs, y_rs)


def rplcc_loss(y_pred, y, eps=1e-8):
    ## Literally (1 - PLCC) / 2
    y_pred, y = gaussian(y_pred), gaussian(y)
    cov = torch.sum(y_pred * y) / y_pred.shape[0]
    # std = (torch.std(y_pred) + eps) * (torch.std(y) + eps)
    return (1 - cov) / 2


def self_similarity_loss(f, f_hat, f_hat_detach=False):
    if f_hat_detach:
        f_hat = f_hat.detach()
    return 1 - torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()


def contrastive_similarity_loss(f, f_hat, f_hat_detach=False, eps=1e-8):
    if f_hat_detach:
        f_hat = f_hat.detach()
    intra_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=1).mean()
    cross_similarity = torch.nn.functional.cosine_similarity(f, f_hat, dim=0).mean()
    return (1 - intra_similarity) / (1 - cross_similarity + eps)