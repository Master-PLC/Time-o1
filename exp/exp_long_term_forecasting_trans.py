import os
import time
import torch
import warnings
import yaml

from copy import deepcopy
import numpy as np
import torch.nn as nn

from exp.exp_basic import Exp_Basic
from utils.dilate_loss import dilate_loss
from utils.dilate_loss_cuda import DilateLossCUDA
from utils.fourier_koopman import fourier_loss
from utils.dtw_cuda import DTW
from utils.metrics import metric
from utils.metrics_torch import create_metric_collector, metric_torch
from utils.polynomial import pca_torch, Basis_Cache, ica_torch, robust_ica_torch, robust_pca_torch, svd_torch, random_torch, Random_Cache, fa_torch
from utils.soft_dtw_cuda import SoftDTW
from utils.tools import EarlyStopping, visual, Scheduler, adjust_learning_rate

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

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()

        eval_time = time.time()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    attn = None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        # total_loss = np.average(total_loss)
        total_loss = torch.mean(torch.stack(total_loss)).item()  # average loss
        self.model.train()
        return total_loss

    def input_transform(self, batch_x, input_cache=None):
        if input_cache is None:
            return batch_x

        if self.args.input_trans == 'None':
            return batch_x
        elif self.args.input_trans == 'random':
            return random_torch(batch_x, self.args.pca_dim, input_cache, self.device)
        elif self.args.input_trans == 'pca':
            return pca_torch(batch_x, self.args.pca_dim, input_cache, self.args.input_use_weights, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'ica':
            return ica_torch(batch_x, self.args.pca_dim, input_cache, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'svd':
            return svd_torch(batch_x, self.args.pca_dim, input_cache, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'robustpca':
            return robust_pca_torch(batch_x, self.args.pca_dim, input_cache, self.args.input_reinit, self.device)
        elif self.args.input_trans == 'robustica':
            return robust_ica_torch(batch_x, self.args.pca_dim, input_cache, self.args.input_reinit, self.device)

    def train(self, setting, prof=None):
        train_data, train_loader = self._get_data(flag='train')
        input_cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'random':
                input_cache = Random_Cache(
                    rank_ratio=self.args.input_rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.seq_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
                random_cache = Random_Cache(
                    rank_ratio=self.args.rank_ratio, pca_dim=self.args.pca_dim, pred_len=self.pred_len, 
                    enc_in=self.args.enc_in, device=self.device
                )
            elif self.args.auxi_type == 'fa':
                input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, mean=train_data.input_mean, device=self.device)
                fa_cache = Basis_Cache(train_data.fa_components, train_data.initializer, mean=train_data.fa_mean, device=self.device)
            elif self.args.auxi_type == 'pca':
                input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, weights=train_data.input_weights, device=self.device)
                pca_cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
            elif self.args.auxi_type == 'robustpca':
                input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, mean=train_data.input_mean, device=self.device)
                robust_pca_cache = Basis_Cache(train_data.pca_components, train_data.initializer, mean=train_data.rpca_mean, device=self.device)
            elif self.args.auxi_type == 'svd':
                input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, device=self.device)
                svd_cache = Basis_Cache(train_data.svd_components, train_data.initializer, device=self.device)
            elif self.args.auxi_type == 'ica':
                input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, mean=train_data.input_mean, whitening=train_data.input_whitening, device=self.device)
                ica_cache = Basis_Cache(train_data.ica_components, train_data.initializer, mean=train_data.ica_mean, whitening=train_data.whitening, device=self.device)
            elif self.args.auxi_type == 'robustica':
                input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, device=self.device)
                robust_ica_cache = Basis_Cache(train_data.ica_components, train_data.initializer, device=self.device)
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
            sdtw = SoftDTW(use_cuda=True, gamma=0.1)
        elif self.args.auxi_mode == 'dtw':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DTW only supports GPU"
            dtw = DTW(use_cuda=True, bandwidth=0.1)
        elif self.args.auxi_mode == 'dilate_cuda':
            assert self.device != 'cpu' and self.device != torch.device('cpu'), "DILATE only supports GPU"
            dilate_cuda = DilateLossCUDA(alpha=self.args.alpha, gamma=self.args.gamma, bandwidth=0)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            has_nan_in_epoch = False
            train_loss = []

            lr_cur = scheduler.get_lr()
            self.writer.add_scalar(f'{self.pred_len}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_x = self.input_transform(batch_x, input_cache)
                batch_y = batch_y.float().to(self.device)

                if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.output_attention:
                    outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    # outputs shape: [B, P, D]
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    attn = None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                loss = 0
                if self.args.rec_lambda:
                    loss_rec = criterion(outputs, batch_y)

                    loss += self.args.rec_lambda * loss_rec

                    if self.step % self.log_step == 0:
                        self.writer.add_scalar(f'{self.pred_len}/train/loss_rec', loss_rec, self.step)

                if self.args.l1_weight:
                    if attn:
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
                                'pca_dim': self.args.pca_dim, 'random_cache': random_cache, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = random_torch(outputs, **kwargs)
                            batch_y = random_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        elif self.args.auxi_type == "fa":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'fa_cache': fa_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = fa_torch(outputs, **kwargs)
                            batch_y = fa_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        elif self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': pca_cache, 'use_weights': self.args.use_weights, 
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = pca_torch(outputs, **kwargs)
                            batch_y = pca_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        elif self.args.auxi_type == "robustpca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': robust_pca_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = robust_pca_torch(outputs, **kwargs)
                            batch_y = robust_pca_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        elif self.args.auxi_type == "svd":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'svd_cache': svd_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = svd_torch(outputs, **kwargs)
                            batch_y = svd_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        elif self.args.auxi_type == "ica":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'ica_cache': ica_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = ica_torch(outputs, **kwargs)
                            batch_y = ica_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        elif self.args.auxi_type == "robustica":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'ica_cache': robust_ica_cache, 'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs = robust_ica_torch(outputs, **kwargs)
                            batch_y = robust_ica_torch(batch_y, **kwargs)
                            loss_auxi = outputs - batch_y
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "fourier_koopman":
                        loss_auxi = fourier_loss(outputs, batch_y, freqs, device=self.device)

                    elif self.args.auxi_mode == "dilate":
                        loss_auxi, _, _ = dilate_loss(outputs, batch_y, self.args.alpha, self.args.gamma, self.device)

                    elif self.args.auxi_mode == "soft_dtw":
                        loss_auxi = sdtw(outputs, batch_y)
                    
                    elif self.args.auxi_mode == "dtw":
                        loss_auxi = dtw(outputs, batch_y)[0].mean()
                        
                    elif self.args.auxi_mode == "dilate_cuda":
                        loss_auxi = dilate_cuda(outputs, batch_y)

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

                    if self.step % self.log_step == 0:
                        self.writer.add_scalar(f'{self.pred_len}/train/loss_auxi', loss_auxi, self.step)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN or Inf, skipping epoch {self.epoch} step {self.step}")
                    has_nan_in_epoch = True
                    continue

                train_loss.append(loss.item())
                self.writer.add_scalar(f'{self.pred_len}/train/loss_iter', loss.item(), self.step)

                if (i + 1) % 100 == 0:
                    print("\titers: {}, epoch: {} | loss: {:.7f}".format(i + 1, self.epoch, loss.item()))
                    cost_time = time.time() - time_now
                    speed = cost_time / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; cost time: {:.4f}s; left time: {:.4f}s'.format(speed, cost_time, left_time))
                    iter_count = 0
                    time_now = time.time()
                    model_state_last_effective = deepcopy(self.model.state_dict())  # save the last effective model state dict

                loss.backward()
                model_optim.step()

                if prof is not None:
                    prof.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

            if model_state_last_effective is not None and has_nan_in_epoch:
                self.model.load_state_dict(model_state_last_effective)

            print("Epoch: {} cost time: {}".format(self.epoch, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            self.writer.add_scalar(f'{self.pred_len}/train/loss', train_loss, self.epoch)
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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        inputs, preds, trues = [], [], []
        folder_path = os.path.join(self.args.test_results, setting)
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        # metric_collector = create_metric_collector(device=self.device)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if ('PEMS' in self.args.data or 'SRU' in self.args.data) and self.args.model not in ['TiDE']:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.output_attention:
                    outputs, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    attn = None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(self.device)

                batch_x = batch_x.detach()
                outputs = outputs.detach()
                batch_y = batch_y.detach()

                if test_data.scale and self.args.inverse:
                    batch_x = batch_x.cpu().numpy()
                    in_shape = batch_x.shape
                    batch_x = test_data.inverse_transform(batch_x.reshape(-1, in_shape[-1])).reshape(in_shape)
                    batch_x = torch.from_numpy(batch_x).float().to(self.device)

                    outputs = outputs.cpu().numpy()
                    batch_y = batch_y.cpu().numpy()
                    out_shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(-1, out_shape[-1])).reshape(out_shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, out_shape[-1])).reshape(out_shape)
                    outputs = torch.from_numpy(outputs).float().to(self.device)
                    batch_y = torch.from_numpy(batch_y).float().to(self.device)

                inputs.append(batch_x)
                preds.append(outputs)
                trues.append(batch_y)

                if i % 20 == 0 and self.output_vis:
                    gt = np.concatenate((batch_x[0, :, -1].cpu().numpy(), batch_y[0, :, -1].cpu().numpy()), axis=0)
                    pd = np.concatenate((batch_x[0, :, -1].cpu().numpy(), outputs[0, :, -1].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        inputs = torch.cat(inputs, dim=0)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)
        print('test shape:', preds.shape, trues.shape)
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        # m = metric_collector.compute()
        # mae, mse, rmse, mape, mspe, mre = m["mae"], m["mse"], m["rmse"], m["mape"], m["mspe"], m["mre"]
        mae, mse, rmse, mape, mspe, mre = metric_torch(preds, trues)
        print('{}\t| mse:{}, mae:{}'.format(self.pred_len, mse, mae))

        self.writer.add_scalar(f'{self.pred_len}/test/mae', mae, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mse', mse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/rmse', rmse, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mape', mape, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mspe', mspe, self.epoch)
        self.writer.add_scalar(f'{self.pred_len}/test/mre', mre, self.epoch)
        self.writer.close()

        log_path = "result_long_term_forecast.txt" if not self.args.log_path else self.args.log_path
        f = open(log_path, 'a')
        f.write(setting + "\n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n\n')
        f.close()

        np.save(os.path.join(res_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe, mre]))

        if self.output_pred:
            np.save(os.path.join(res_path, 'input.npy'), inputs.cpu().numpy())
            np.save(os.path.join(res_path, 'pred.npy'), preds.cpu().numpy())
            np.save(os.path.join(res_path, 'true.npy'), trues.cpu().numpy())
            if self.args.auxi_mode == 'basis' and self.args.auxi_type == 'pca':
                train_data, _ = self._get_data(flag='train')
                pca_components = train_data.pca_components
                np.save(os.path.join(res_path, 'pca_components.npy'), pca_components)

        print('save configs')
        args_dict = vars(self.args)
        with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
            yaml.dump(args_dict, yaml_file, default_flow_style=False)

        return
