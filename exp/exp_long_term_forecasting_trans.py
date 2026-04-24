import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import time
import torch
import warnings

from copy import deepcopy
import numpy as np
import torch.nn as nn
from tslearn.metrics import dtw as dtw2, dtw_limited_warping_length

from exp.exp_basic import Exp_Basic
from utils.dilate_loss import dilate_loss
from utils.dilate_loss_cuda import DilateLossCUDA
from utils.dpp_loss import dpp_loss
from utils.dtw_cuda import DTW
from utils.fourier_koopman import fourier_loss
from utils.gdtw_cuda import GromovDTW
from utils.ldtw_cuda import LDTW
from utils.polynomial import (pca_torch, Basis_Cache, ica_torch, robust_ica_torch, robust_pca_torch, svd_torch, random_torch, Random_Cache, fa_torch,
                              pca_torch_inverse, ica_torch_inverse, robust_ica_torch_inverse, robust_pca_torch_inverse, svd_torch_inverse, random_torch_inverse, fa_torch_inverse)
from utils.soft_dtw_cuda import SoftDTW
from utils.tools import EarlyStopping, Scheduler

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast_Trans(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.label_len = args.label_len

        self.input_rank_ratio = args.input_rank_ratio
        if self.input_rank_ratio and self.input_rank_ratio <= 1.0:
            seq_len = int(self.seq_len * self.input_rank_ratio)
        elif self.input_rank_ratio < 0 or self.input_rank_ratio > 1:
            seq_len = int(abs(args.input_rank_ratio))
            seq_len = min(seq_len, self.seq_len)
        self.input_seq_len = seq_len
        
        self.rank_ratio = args.rank_ratio
        if self.rank_ratio and self.rank_ratio <= 1.0:
            pred_len = int(self.pred_len * self.rank_ratio)
        elif self.rank_ratio < 0 or self.rank_ratio > 1:
            pred_len = int(abs(self.rank_ratio))
            pred_len = min(pred_len, self.pred_len)
        self.output_pred_len = pred_len

    def _build_model(self):
        args = deepcopy(self.args)
        args.seq_len = self.input_seq_len
        args.pred_len = self.output_pred_len
        model = self.model_dict[args.model].Model(args).float()

        pretrain_model_path = args.pretrain_model_path
        if pretrain_model_path and os.path.exists(pretrain_model_path):
            print(f'Loading pretrained model from {pretrain_model_path}')
            state_dict = torch.load(pretrain_model_path)
            model.load_state_dict(state_dict, strict=False)

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def initialize_cache(self, train_data):
        self.input_cache = None
        self.output_cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'random':
                self.input_cache = Random_Cache(
                    rank_ratio=self.args.input_rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.seq_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
                self.output_cache = Random_Cache(
                    rank_ratio=self.args.rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.pred_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
            elif self.args.auxi_type == 'fa':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, mean=train_data.input_mean, device=self.device)
                self.output_cache = Basis_Cache(train_data.fa_components, train_data.initializer, mean=train_data.fa_mean, device=self.device)
            elif self.args.auxi_type == 'pca':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, weights=train_data.input_weights, device=self.device)
                self.output_cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
            elif self.args.auxi_type == 'robustpca':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, mean=train_data.input_mean, device=self.device)
                self.output_cache = Basis_Cache(train_data.pca_components, train_data.initializer, mean=train_data.rpca_mean, device=self.device)
            elif self.args.auxi_type == 'svd':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, device=self.device)
                self.output_cache = Basis_Cache(train_data.svd_components, train_data.initializer, device=self.device)
            elif self.args.auxi_type == 'ica':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, mean=train_data.input_mean, whitening=train_data.input_whitening, device=self.device)
                self.output_cache = Basis_Cache(train_data.ica_components, train_data.initializer, mean=train_data.ica_mean, whitening=train_data.whitening, device=self.device)
            elif self.args.auxi_type == 'robustica':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, device=self.device)
                self.output_cache = Basis_Cache(train_data.ica_components, train_data.initializer, device=self.device)

    def input_transform(self, batch_x):
        if self.input_cache is None:
            return batch_x

        if self.args.input_trans == 'None':
            return batch_x
        elif self.args.input_trans == 'random':
            return random_torch(batch_x, self.args.pca_dim, self.input_cache, self.device)
        elif self.args.input_trans == 'pca':
            return pca_torch(batch_x, self.args.pca_dim, self.input_cache, self.args.input_use_weights, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'ica':
            return ica_torch(batch_x, self.args.pca_dim, self.input_cache, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'svd':
            return svd_torch(batch_x, self.args.pca_dim, self.input_cache, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'robustpca':
            return robust_pca_torch(batch_x, self.args.pca_dim, self.input_cache, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'robustica':
            return robust_ica_torch(batch_x, self.args.pca_dim, self.input_cache, self.args.input_reinit, self.device)

    def output_transform(self, outputs):
        if self.output_cache is None:
            return outputs

        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == "random":
                return random_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.pred_len, self.device)
            elif self.args.auxi_type == "fa":
                return fa_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.reinit, self.pred_len, self.device)
            elif self.args.auxi_type == "pca":
                return pca_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.use_weights, self.args.reinit, self.pred_len, self.device)
            elif self.args.auxi_type == "robustpca":
                return robust_pca_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.reinit, self.pred_len, self.device)
            elif self.args.auxi_type == "svd":
                return svd_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.reinit, self.pred_len, self.device)
            elif self.args.auxi_type == "ica":
                return ica_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.reinit, self.pred_len, self.device)
            elif self.args.auxi_type == "robustica":
                return robust_ica_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.reinit, self.pred_len, self.device)
        else:
            return outputs

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark, return_trans=False):
        batch_x = batch_x.float().to(self.device)
        batch_x = self.input_transform(batch_x)
        batch_y = batch_y.float().to(self.device)

        if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE', 'CFPT']:
            batch_x_mark = None
            batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, batch_x_mark, dec_inp, batch_y_mark]
        if self.args.output_attention:
            outputs, attn = self.model(*model_args)
        else:
            outputs = self.model(*model_args)
            attn = None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs_trans = outputs[:, -self.output_pred_len:, f_dim:]
        outputs_time = self.output_transform(outputs_trans)
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        if return_trans:
            return outputs_trans, outputs_time, batch_y, attn
        return outputs_time, batch_y, attn

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        self.initialize_cache(train_data)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        model_state_last_effective = None
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        if self.args.auxi_mode == 'fourier_koopman':
            freqs = nn.Parameter(torch.tensor(train_data.freqs, device=self.device, dtype=torch.float32))
            model_optim.add_param_group({'params': freqs, 'lr': self.args.learning_rate})
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion()
        if self.args.auxi_mode == 'soft_dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "SoftDTW only supports GPU"
            sdtw = SoftDTW(use_cuda=True, gamma=self.args.gamma, bandwidth=self.args.bandwidth)
        elif self.args.auxi_mode == 'dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DTW only supports GPU"
            dtw = DTW(use_cuda=True, bandwidth=self.args.bandwidth)
        elif self.args.auxi_mode == 'ldtw2':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "LDTW only supports GPU"
            ldtw = LDTW(use_cuda=True, bandwidth=self.args.bandwidth, max_length=self.args.warping_length)
        elif self.args.auxi_mode == 'dilate_cuda':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DILATE only supports GPU"
            dilate_cuda = DilateLossCUDA(alpha=self.args.dilate_alpha, gamma=self.args.gamma, bandwidth=self.args.bandwidth)
        elif self.args.auxi_mode == 'gdtw':
            gdtw = GromovDTW(
                max_iter=self.args.max_iter, gamma=self.args.gamma, solver='soft', bandwidth=self.args.bandwidth,
                tol=self.args.stopThr, device=self.device
            )

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            has_nan_in_epoch = False
            train_loss = []
            rec_loss, auxi_loss = [], []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs_trans, outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark, return_trans=True)

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, batch_y)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)

                if self.args.l1_weight and attn:
                    loss += self.args.l1_weight * attn[0]

                if self.args.auxi_lambda:
                    if self.args.joint_forecast:  # joint distribution forecasting
                        outputs = torch.concat((batch_x, outputs), dim=1)  # [B, S+P, D]
                        batch_y = torch.concat((batch_x, batch_y), dim=1)  # [B, S+P, D]

                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # shape: [B, P, D]

                    elif self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "basis":
                        kwargs = {'degree': self.args.leg_degree, 'device': self.device}
                        if self.args.auxi_type == "random":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'random_cache': self.output_cache, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = random_torch(outputs, **kwargs)
                            batch_y = random_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        elif self.args.auxi_type == "fa":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'fa_cache': self.output_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = fa_torch(outputs, **kwargs)
                            batch_y = fa_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        elif self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': self.output_cache, 'use_weights': self.args.use_weights, 
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = pca_torch(outputs, **kwargs)
                            batch_y = pca_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        elif self.args.auxi_type == "robustpca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': self.output_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = robust_pca_torch(outputs, **kwargs)
                            batch_y = robust_pca_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        elif self.args.auxi_type == "svd":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'svd_cache': self.output_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = svd_torch(outputs, **kwargs)
                            batch_y = svd_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        elif self.args.auxi_type == "ica":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'ica_cache': self.output_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = ica_torch(outputs, **kwargs)
                            batch_y = ica_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        elif self.args.auxi_type == "robustica":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'ica_cache': self.output_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = robust_ica_torch(outputs, **kwargs)
                            batch_y = robust_ica_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "fourier_koopman":
                        loss_auxi = fourier_loss(outputs, batch_y, freqs, device=self.device)

                    elif self.args.auxi_mode == "dilate":
                        loss_auxi, _, _ = dilate_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "dpp":
                        loss_auxi = dpp_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "soft_dtw":
                        loss_auxi = sdtw(outputs, batch_y)
                    
                    elif self.args.auxi_mode == "dtw":
                        loss_auxi = dtw(outputs, batch_y)[0].mean()

                    elif self.args.auxi_mode == "ldtw2":
                        loss_auxi = ldtw(outputs, batch_y)[0].mean()

                    elif self.args.auxi_mode == "dtw2":
                        loss_auxi = dtw2(
                            outputs.permute(1, 0, 2).reshape(self.pred_len, -1),
                            batch_y.permute(1, 0, 2).reshape(self.pred_len, -1),
                        )

                    elif self.args.auxi_mode == "ldtw":
                        loss_auxi = dtw_limited_warping_length(
                            outputs.permute(1, 0, 2).reshape(self.pred_len, -1),
                            batch_y.permute(1, 0, 2).reshape(self.pred_len, -1),
                            max_length=self.args.warping_length
                        )

                    elif self.args.auxi_mode == "dilate_cuda":
                        loss_auxi = dilate_cuda(outputs, batch_y)

                    elif self.args.auxi_mode == "gdtw":
                        loss_auxi = gdtw(outputs, batch_y)

                    else:
                        raise NotImplementedError

                    if self.args.auxi_loss == "MAE":
                        # MAE, 最小化element-wise error的模长
                        loss_auxi = loss_auxi.abs().mean() if self.args.module_first else loss_auxi.mean().abs()  # check the dim of fft
                    elif self.args.auxi_loss == "MSE":
                        # MSE, 最小化element-wise error的模长
                        loss_auxi = (loss_auxi.abs()**2).mean() if self.args.module_first else (loss_auxi**2).mean().abs()
                    elif self.args.auxi_loss == "None":
                        pass
                    else:
                        raise NotImplementedError

                    loss += self.args.auxi_lambda * loss_auxi
                else:
                    loss_auxi = torch.tensor(1e4)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN or Inf, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                train_loss.append(loss.item())
                rec_loss.append(loss_rec.item())
                auxi_loss.append(loss_auxi.item())
                self.writer.add_scalar(f'{self.pred_len}/train_iter/loss', loss.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_rec', loss_rec.item(), self.step)
                self.writer.add_scalar(f'{self.pred_len}/train_iter/loss_auxi', loss_auxi.item(), self.step)

                if (i + 1) % 100 == 0:
                    print("\titers: {}, epoch: {} | loss_rec: {:.7f}, loss_auxi: {:.7f}, loss: {:.7f}".format(i + 1, self.epoch, loss_rec.item(), loss_auxi.item(), loss.item()))
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                loss.backward()
                model_optim.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            rec_loss = np.average(rec_loss)
            auxi_loss = np.average(auxi_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', rec_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', auxi_loss, self.epoch)
            self.writer.add_scalar(f'{self.pred_len}/vali/loss', vali_loss, self.epoch)

            print(
                "Epoch: {}, Steps: {} | Train Loss: {:.7f} Vali Loss: {:.7f}".format(
                    self.epoch, self.step, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj not in ['TST']:
                scheduler.step(vali_loss, self.epoch)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
