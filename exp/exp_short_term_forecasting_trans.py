import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import pandas
import time
import torch
import warnings
import yaml

from copy import deepcopy
import numpy as np
import torch.nn as nn

from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.m4_summary import M4Summary
from utils.polynomial import pca_torch, Basis_Cache, pca_torch_inverse
from utils.tools import EarlyStopping, Scheduler, visual

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast_Trans(Exp_Basic):
    def __init__(self, args):
        self.seasonal_patterns = args.seasonal_patterns
        if 'm4' in args.data:
            args.pred_len = M4Meta.horizons_map[self.seasonal_patterns]  # Up to M4 config
            args.seq_len = 2 * args.pred_len  # input_len = 2*pred_len
            args.label_len = args.pred_len
            args.frequency_map = M4Meta.frequency_map[self.seasonal_patterns]

        super().__init__(args)
        
        self.seq_len = self.args.seq_len
        self.pred_len = self.args.pred_len
        self.label_len = self.args.label_len
        
        self.input_rank_ratio = args.input_rank_ratio
        if self.input_rank_ratio and self.input_rank_ratio <= 1.0:
            seq_len = int(self.seq_len * self.input_rank_ratio)
        elif self.input_rank_ratio < 0 or self.input_rank_ratio > 1:
            seq_len = int(abs(self.input_rank_ratio))
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

        if args.use_multi_gpu and args.use_gpu:
            model = nn.DataParallel(model, device_ids=args.device_ids)
        return model

    def input_transform(self, batch_x):
        if self.input_cache is None:
            return batch_x

        if self.args.input_trans == 'None':
            return batch_x
        elif self.args.input_trans == 'pca':
            return pca_torch(batch_x, self.args.pca_dim, self.input_cache, self.args.input_use_weights, self.args.input_reinit, self.device)

    def output_transform(self, outputs):
        if self.output_cache is None:
            return outputs

        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'pca':
                return pca_torch_inverse(outputs, self.args.pca_dim, self.output_cache, self.args.use_weights, self.args.reinit, self.pred_len, self.device)
        else:
            return outputs

    def forward_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_x = self.input_transform(batch_x)
        batch_y = batch_y.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        model_args = [batch_x, None, dec_inp, None]
        if self.args.output_attention:
            outputs, attn = self.model(*model_args)
        else:
            outputs = self.model(*model_args)
            attn = None

        f_dim = -1 if self.args.features == 'MS' else 0
        outputs_trans = outputs[:, -self.output_pred_len:, f_dim:]
        outputs_time = self.output_transform(outputs_trans)
        batch_y = batch_y[:, -self.pred_len:, f_dim:]
        batch_y_mark = batch_y_mark[:, -self.pred_len:, f_dim:].to(self.device)

        return outputs_trans, outputs_time, batch_y, attn, batch_y_mark

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        self.input_cache = None
        self.output_cache = None
        if self.args.auxi_mode == 'basis':
            if self.args.auxi_type == 'pca':
                self.input_cache = Basis_Cache(train_data.input_components, train_data.input_initializer, weights=train_data.input_weights, device=self.device)
                self.output_cache = Basis_Cache(train_data.pca_components, train_data.initializer, weights=train_data.weights, device=self.device)
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        self.writer = self._create_writer(res_path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = Scheduler(model_optim, self.args, train_steps)
        criterion = self._select_criterion(self.args.loss)

        for epoch in range(self.args.train_epochs):
            self.epoch = epoch + 1
            iter_count = 0
            train_loss = []
            rec_loss, auxi_loss = [], []

            lr_cur = scheduler.get_lr()
            lr_cur = lr_cur[0] if isinstance(lr_cur, list) else lr_cur
            self.writer.add_scalar(f'{self.seasonal_patterns}/train/lr', lr_cur, self.epoch)

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                self.step += 1
                iter_count += 1
                model_optim.zero_grad()

                outputs_trans, outputs, batch_y, attn, batch_y_mark = self.forward_step(batch_x, batch_y, batch_x_mark, batch_y_mark)

                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))

                loss = 0  # + loss_sharpness * 1e-5
                if self.args.rec_lambda:
                    loss_rec = criterion(batch_x.to(self.device), self.args.frequency_map, outputs, batch_y, batch_y_mark)
                    loss += self.args.rec_lambda * loss_rec
                else:
                    loss_rec = torch.tensor(1e4)

                if self.args.auxi_lambda:
                    if self.args.auxi_mode == "fft":
                        loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y, dim=1)  # shape: [B, P, D]

                    elif self.args.auxi_mode == "rfft":
                        if self.args.auxi_type == 'complex':
                            loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y, dim=1)  # shape: [B, P//2+1, D]
                        else:
                            raise NotImplementedError

                    elif self.args.auxi_mode == "basis":
                        kwargs = {'degree': self.args.leg_degree, 'device': self.device}
                        if self.args.auxi_type == "pca":
                            kwargs = {
                                'pca_dim': self.args.pca_dim, 'pca_cache': self.output_cache, 'use_weights': self.args.use_weights, 
                                'reinit': self.args.reinit, 'device': self.device
                            }
                            if self.args.input_trans == 'None':
                                outputs_trans = pca_torch(outputs, **kwargs)
                            batch_y = pca_torch(batch_y, **kwargs)
                            loss_auxi = outputs_trans - batch_y
                        else:
                            raise NotImplementedError

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

                loss.backward()
                model_optim.step()

                if self.args.lradj in ['TST']:
                    scheduler.step(verbose=(i + 1 == train_steps))

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

    def vali(self, train_loader, vali_loader, criterion):
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # shape [B, S, 1]

        self.model.eval()
        eval_time = time.time()
        with torch.no_grad():
            # decoder input
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.label_len:, :], dec_inp], dim=1).float()

            # encoder - decoder
            outputs = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 500)  # validation set size
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                batch_x = self.input_transform(batch_x)
                outputs_trans = self.model(
                    batch_x, None, dec_inp[id_list[i]:id_list[i + 1]], None
                ).detach()
                outputs_time = self.output_transform(outputs_trans)
                outputs[id_list[i]:id_list[i + 1], :, :] = outputs_time

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.pred_len:, f_dim:]

            pred = outputs
            true = torch.from_numpy(np.array(y)).to(self.device)
            batch_y_mark = torch.ones(true.shape).to(self.device)

            loss = criterion(x.detach()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        print('Validation cost time: {}'.format(time.time() - eval_time))
        self.model.train()
        return loss

    def test(self, setting, test=0):
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')

        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1)  # shape [B, S, 1]

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        if self.output_vis:
            folder_path = os.path.join(self.args.test_results, setting)
            os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            B, _, C = x.shape
            dec_inp = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.label_len:, :], dec_inp], dim=1).float()

            # encoder - decoder
            outputs = torch.zeros((B, self.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1)
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_x = x[id_list[i]:id_list[i + 1]]
                batch_x = self.input_transform(batch_x)
                outputs_trans = self.model(batch_x, None, dec_inp[id_list[i]:id_list[i + 1]], None)
                outputs_time = self.output_transform(outputs_trans)
                outputs[id_list[i]:id_list[i + 1], :, :] = outputs_time

                if id_list[i] % 1000 == 0:
                    print(id_list[i])

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.pred_len:, f_dim:]
            outputs = outputs.detach()

            preds = outputs
            trues = torch.from_numpy(np.array(y)).to(self.device)
            x = x.detach()

            if self.output_vis:
                for i in range(0, preds.shape[0], preds.shape[0] // 10):
                    gt = np.concatenate((x[i, :, 0].cpu().numpy(), trues[i].cpu().numpy()), axis=0)
                    pd = np.concatenate((x[i, :, 0].cpu().numpy(), preds[i, :, 0].cpu().numpy()), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        print('test shape:', preds.shape, trues.shape)

        # result save
        res_path = os.path.join(self.args.results, setting)
        os.makedirs(res_path, exist_ok=True)
        summary_path = os.path.join(self.args.results, "m4_results")
        os.makedirs(summary_path, exist_ok=True)
        if self.writer is None:
            self.writer = self._create_writer(res_path)

        forecasts_df = pandas.DataFrame(preds[:, :, 0].cpu().numpy(), columns=[f'V{i+1}' for i in range(self.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        forecasts_df.set_index(forecasts_df.columns[0], inplace=True)
        forecasts_df.to_csv(os.path.join(summary_path, self.seasonal_patterns + '_forecast.csv'))

        filenames = os.listdir(summary_path)
        if 'Weekly_forecast.csv' in filenames \
            and 'Monthly_forecast.csv' in filenames \
            and 'Yearly_forecast.csv' in filenames \
            and 'Daily_forecast.csv' in filenames \
            and 'Hourly_forecast.csv' in filenames \
            and 'Quarterly_forecast.csv' in filenames:
            m4_summary = M4Summary(summary_path, self.args.root_path)
            # m4_forecast.set_index(m4_winner_forecast.columns[0], inplace=True)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            metrics = {
                "smape": smape_results,
                "mape": mape,
                "mase": mase,
                "owa": owa_results
            }
            line = ''
            for k, v in metrics.items():
                line += f"{k}: {v}\n"
            print(line)

            yaml.safe_dump(metrics, open(os.path.join(summary_path, "metrics.yaml"), 'w'), default_flow_style=False, sort_keys=False)

            if self.output_log:
                log_path = "result_short_term_forecast.txt" if not self.args.log_path else self.args.log_path
                payload = f"{setting}\n\n{line}\n\n"
                with open(log_path, mode="a", encoding="utf-8") as f:
                    f.write(payload)

        else:
            print('After all 6 tasks are finished, you can calculate the averaged index')

        self.writer.close()

        print('save configs')
        yaml.dump(vars(self.args), open(os.path.join(res_path, 'config.yaml'), 'w'), default_flow_style=False)

        return
