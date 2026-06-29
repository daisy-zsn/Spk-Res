import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.ewc import EWC
from basicsr.models.kd_loss import KDLoss

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import numpy as np
import cv2
import torch.nn.functional as F
from functools import partial

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

class ReconstructionModel(BaseModel):
    def __init__(self, opt):
        super(ReconstructionModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        logger = get_root_logger()

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # ========== L2-SP: save pretrained parameter snapshot ==========
        self.l2_sp_weight = train_opt.get('l2_sp_weight', 0.0)
        if self.l2_sp_weight > 0:
            self.pretrained_params = {}
            for name, param in self.net_g.named_parameters():
                self.pretrained_params[name] = param.detach().clone()
            logger.info(f'L2-SP enabled with weight: {self.l2_sp_weight}')

        # ========== EWC: initialize Elastic Weight Consolidation ==========
        self.ewc_lambda = train_opt.get('ewc_lambda', 0.0)
        self.ewc = None
        if self.ewc_lambda > 0:
            self.ewc = EWC(self.net_g, self.ewc_lambda)
            logger.info(f'EWC enabled with lambda: {self.ewc_lambda}')

        # ========== Knowledge Distillation: freeze teacher model ==========
        self.kd_weight = train_opt.get('kd_weight', 0.0)
        self.kd_feature_weight = train_opt.get('kd_feature_weight', 0.0)
        self.teacher_model = None
        if (self.kd_weight > 0 or self.kd_feature_weight > 0) and self.opt['path'].get('pretrain_network_g', None) is not None:
            self.teacher_model = define_network(deepcopy(self.opt['network_g']))
            self.teacher_model = self.model_to_device(self.teacher_model)
            load_path = self.opt['path']['pretrain_network_g']
            self.load_network(self.teacher_model, load_path,
                              self.opt['path'].get('strict_load_g', True),
                              param_key=self.opt['path'].get('param_key', 'params'))
            # freeze teacher
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            # init KD loss
            kd_temperature = train_opt.get('kd_temperature', 4.0)
            self.kd_loss = KDLoss(temperature=kd_temperature)
            logger.info(f'KD enabled: output_weight={self.kd_weight}, feature_weight={self.kd_feature_weight}, temperature={kd_temperature}')

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        # ========== Layer-wise LR Decay ==========
        layer_lr_decay = train_opt.get('layer_lr_decay', 1.0)
        logger_l = get_root_logger()

        if layer_lr_decay < 1.0:
            # 按深度分组：浅层 low lr，深层 high lr
            param_list = list(self.net_g.named_parameters())
            num_layers = len(param_list)
            for idx, (k, v) in enumerate(param_list):
                if v.requires_grad:
                    depth_ratio = idx / max(num_layers - 1, 1)
                    # 浅层(head, body前端)用较小lr，深层(body后端, tail)用较大lr
                    lr_scale = layer_lr_decay ** ((1.0 - depth_ratio) * 10)
                    optim_params.append({'params': v, 'lr': train_opt['optim_g']['lr'] * lr_scale})
                else:
                    logger_l.warning(f'Params {k} will not be optimized.')
            logger_l.info(f'Layer-wise LR Decay enabled: decay_rate={layer_lr_decay}')
        else:
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    logger_l.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        # 如果是 layer-wise 模式，optim_g 中的 lr 会被 param_groups 覆盖，所以 pop 掉
        optim_kwargs = {}
        for k_opt in list(train_opt['optim_g'].keys()):
            if k_opt != 'lr' or layer_lr_decay == 1.0:
                optim_kwargs[k_opt] = train_opt['optim_g'][k_opt]

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **optim_kwargs)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **optim_kwargs)
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        # ========== Register KD feature hooks before forward (if needed) ==========
        student_feat = None
        student_hook_handle = None
        if self.teacher_model is not None and self.kd_feature_weight > 0:
            def _student_hook(module, input, output):
                nonlocal student_feat
                student_feat = output
            # EDSR: hook on .body; Restormer/SpkRes: hook on .refinement
            if hasattr(self.net_g, 'body'):
                student_hook_handle = self.net_g.body.register_forward_hook(_student_hook)
            elif hasattr(self.net_g, 'refinement'):
                student_hook_handle = self.net_g.refinement.register_forward_hook(_student_hook)

        preds = self.net_g(self.lq)

        if student_hook_handle is not None:
            student_hook_handle.remove()

        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        for pred in preds:
            l_pix += self.cri_pix(pred, self.gt)

        loss_dict['l_pix'] = l_pix

        l_total = l_pix

        # ========== KL divergence (for VAE-based models) ==========
        kl_weight = self.opt['train'].get('kl_weight', 0)
        if kl_weight > 0 and hasattr(self.net_g, 'kl_loss') and self.net_g.kl_loss is not None:
            l_kl = kl_weight * self.net_g.kl_loss
            loss_dict['l_kl'] = l_kl
            l_total += l_kl # add kl loss

        # ========== L2-SP: Simple Parameter Penalty ==========
        if self.l2_sp_weight > 0:
            l2_sp = 0.
            for name, param in self.net_g.named_parameters():
                l2_sp += torch.sum((param - self.pretrained_params[name].to(param.device)) ** 2)
            l_sp = self.l2_sp_weight * l2_sp
            loss_dict['l_sp'] = l_sp
            l_total += l_sp

        # ========== EWC: Elastic Weight Consolidation ==========
        if self.ewc is not None and self.ewc_lambda > 0:
            ewc_loss = self.ewc.penalty(self.net_g)
            loss_dict['l_ewc'] = ewc_loss
            l_total += ewc_loss

        # ========== Knowledge Distillation ==========
        if self.teacher_model is not None:
            # Teacher forward (with feature hook if kd_feature_weight > 0)
            teacher_feat = None
            teacher_hook_handle = None
            with torch.no_grad():
                if self.kd_feature_weight > 0:
                    def _teacher_hook(module, input, output):
                        nonlocal teacher_feat
                        teacher_feat = output
                    # Match architecture: EDSR -> .body, Restormer/SpkRes -> .refinement
                    if hasattr(self.teacher_model, 'body'):
                        teacher_hook_handle = self.teacher_model.body.register_forward_hook(_teacher_hook)
                    elif hasattr(self.teacher_model, 'refinement'):
                        teacher_hook_handle = self.teacher_model.refinement.register_forward_hook(_teacher_hook)

                teacher_output = self.teacher_model(self.lq)

                if teacher_hook_handle is not None:
                    teacher_hook_handle.remove()

                if isinstance(teacher_output, list):
                    teacher_output = teacher_output[-1]

            # Output KD loss
            if self.kd_weight > 0:
                l_kd = self.kd_weight * self.kd_loss(self.output, teacher_output)
                loss_dict['l_kd'] = l_kd
                l_total += l_kd

            # Feature KD loss (MSE between intermediate features)
            if self.kd_feature_weight > 0:
                if student_feat is not None and teacher_feat is not None:
                    l_fkd = self.kd_feature_weight * F.mse_loss(student_feat, teacher_feat)
                    loss_dict['l_fkd'] = l_fkd
                    l_total += l_fkd

        # Backward
        l_total.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01) # 梯度裁剪，防止梯度爆炸
        self.optimizer_g.step() # AdamW

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def nonpad_test(self, img=None):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                pred = self.net_g_ema(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        else:
            self.net_g.eval()
            with torch.no_grad():
                pred = self.net_g(img)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            # 验证方法
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        if metric_type == 'calculate_psnr':
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                        elif metric_type == 'calculate_mse':
                            self.metric_results[name] += self.cri_pix(visuals['result'], visuals['gt'])  # MSE
                else:
                    for name, opt_ in opt_metric.items():
                        # calculate_mse
                        # metric_type = opt_.pop('type')
                        # self.metric_results[name] += getattr(
                        #     metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

                        metric_type = opt_.pop('type')
                        if metric_type == 'calculate_psnr':
                            self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)
                        elif metric_type == 'calculate_mse':
                            self.metric_results[name] += self.cri_pix(visuals['result'], visuals['gt']) # MSE
            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
    
    def save_best_weights(self, weights_path):
        self.save_weights(self.net_g, weights_path)
