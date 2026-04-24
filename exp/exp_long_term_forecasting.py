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
from utils.polynomial import pca_torch, Basis_Cache, ica_torch, robust_ica_torch, robust_pca_torch, svd_torch, random_torch, Random_Cache, fa_torch
from utils.soft_dtw_cuda import SoftDTW
from utils.tools import EarlyStopping, Scheduler

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.pred_len = args.pred_len
        self.label_len = args.label_len

    def initialize_cache(self, train_data):
        cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'random':
                cache = Random_Cache(
                    rank_ratio=self.args.rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.pred_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
            elif self.args.auxi_type == 'fa':
                cache = Basis_Cache(train_data.fa_components, train_data.initializer, mean=train_data.fa_mean, device=self.device)
            elif self.args.auxi_type == 'pca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
            elif self.args.auxi_type == 'robustpca':
                cache = Basis_Cache(train_data.pca_components, train_data.initializer, mean=train_data.rpca_mean, device=self.device)
            elif self.args.auxi_type == 'svd':
                cache = Basis_Cache(train_data.svd_components, train_data.initializer, device=self.device)
            elif self.args.auxi_type == 'ica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, mean=train_data.ica_mean, whitening=train_data.whitening, device=self.device)
            elif self.args.auxi_type == 'robustica':
                cache = Basis_Cache(train_data.ica_components, train_data.initializer, device=self.device)
        return cache

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        cache = self.initialize_cache(train_data)
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

                outputs, batch_y, attn = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

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
                        outputs = torch.concat((batch_x.to(outputs.device), outputs), dim=1).float()  # [B, S+P, D]
                        batch_y = torch.concat((batch_x.to(batch_y.device), batch_y), dim=1).float()  # [B, S+P, D]

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
                                'pca_dim': self.args.pca_dim, 'random_cache': cache, 'device': self.device
                            }
                            loss_auxi = random_torch(outputs, **kwargs) - random_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "fa":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'fa_cache': cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = fa_torch(outputs, **kwargs) - fa_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'use_weights': self.args.use_weights, 
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = pca_torch(outputs, **kwargs) - pca_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "robustpca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = robust_pca_torch(outputs, **kwargs) - robust_pca_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "svd":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'svd_cache': cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = svd_torch(outputs, **kwargs) - svd_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "ica":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'ica_cache': cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = ica_torch(outputs, **kwargs) - ica_torch(batch_y, **kwargs)
                        elif self.args.auxi_type == "robustica":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'ica_cache': cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            loss_auxi = robust_ica_torch(outputs, **kwargs) - robust_ica_torch(batch_y, **kwargs)
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
