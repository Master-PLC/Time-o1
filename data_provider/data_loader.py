import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import glob
import os
import re
import torch
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset

from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import Normalizer, interpolate_missing, subsample
from utils.fourier_koopman import fourier
from utils.polynomial import get_pca_base, get_cca_projection, get_ica_base, get_robustica_base, get_robustpca_base, get_svd_base, get_fa_base
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour_Fourier(Dataset_ETT_hour):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., num_freqs=16, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.fourier_fit(num_freqs)

    def fourier_fit(self, num_freqs=10):
        if self.set_type != 0:
            self.freqs = None
            return

        print("Fitting fourier ...")
        f = fourier(num_freqs=num_freqs)
        f.fft(self.data_x)
        self.freqs = f.freqs
        print(f"Fourier freqs: {self.freqs}")


class Dataset_ETT_hour_Trend(Dataset_ETT_hour):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., trend_k=0.02, **kwargs
    ):
        self.trend_k = trend_k

        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        data_len = len(df_data)
        trend = np.arange(data_len) * self.trend_k
        df_data += trend[:, None]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


class Dataset_ETT_hour_PCA(Dataset_ETT_hour):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., trend_k=0.02, rank_ratio=1.0, 
        input_rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0, load_from_disk="", **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.load_from_disk = load_from_disk
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        self.pca_components = None
        self.input_components = None

        if self.set_type != 0:
            return

        print("Fitting PCA ...")
        if self.load_from_disk:
            proj_dir = os.path.join(self.load_from_disk, f"sp{self.speedup_sklearn}")
            if os.path.exists(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy')):
                self.input_components = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'))
                self.input_initializer = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'))
                self.input_weights = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'))
            if os.path.exists(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy')):
                self.pca_components = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'))
                self.initializer = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'))
                self.weights = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'))

        if self.pca_components is None and self.input_components is None:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                label = label[-self.pred_len:]
                input_seq.append(inp)
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
            self.input_components, self.input_initializer, self.input_weights = get_pca_base(input_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            self.pca_components, self.initializer, self.weights = get_pca_base(label_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            if self.load_from_disk:
                os.makedirs(os.path.join(proj_dir, 'input', f"{self.seq_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'), self.input_components)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'), self.input_initializer)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'), self.input_weights)
                os.makedirs(os.path.join(proj_dir, 'output', f"{self.pred_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'), self.pca_components)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'), self.initializer)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'), self.weights)

        if input_rank_ratio and input_rank_ratio <= 1.0:
            full_rank = self.input_weights.shape[-1]
            proj_dim = int(full_rank * input_rank_ratio)
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]
        elif input_rank_ratio < 0 or input_rank_ratio > 1:
            proj_dim = int(abs(input_rank_ratio))
            proj_dim = min(proj_dim, self.input_weights.shape[-1])
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]

        print(f"Input PCA components shape: {self.input_components.shape}")
        print(f"Input PCA weights shape: {self.input_weights.shape}")

        if rank_ratio and rank_ratio <= 1.0:
            full_rank = self.weights.shape[-1]
            proj_dim = int(full_rank * rank_ratio)
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]
        elif rank_ratio < 0 or rank_ratio > 1:
            proj_dim = int(abs(rank_ratio))
            proj_dim = min(proj_dim, self.weights.shape[-1])
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]

        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")


class Dataset_ETT_hour_CCA(Dataset_ETT_hour):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., trend_k=0.02, rank_ratio=1.0, 
        input_rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0, align_type=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.align_type = align_type
        self.speedup_sklearn = speedup_sklearn
        self.cca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def cca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="D", reinit=0):
        if self.set_type != 0:
            self.Wx = None
            self.Wy = None
            return

        print("Fitting CCA ...")
        if self.align_type != 5:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                input_seq.append(inp)
                label = label[-self.pred_len:]
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
        elif self.align_type == 5:
            input_seq = self.data_x[:-self.pred_len]  # shape: [N, D]
            label_seq = self.data_y[self.pred_len:]  # shape: [N, D]
        self.Wx, self.Wy, self.means, self.stds = get_cca_projection(
            input_seq, label_seq, rank_ratio, pca_dim, self.speedup_sklearn, self.align_type
        )
        print(f"CCA Wx shape: {self.Wx.shape}")
        print(f"CCA Wy shape: {self.Wy.shape}")


class Dataset_ETT_minute(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv',
        target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute_Fourier(Dataset_ETT_minute):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTm1.csv',
        target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., num_freqs=16, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.fourier_fit(num_freqs)

    def fourier_fit(self, num_freqs=10):
        if self.set_type != 0:
            self.freqs = None
            return

        print("Fitting fourier ...")
        f = fourier(num_freqs=num_freqs)
        f.fft(self.data_x)
        self.freqs = f.freqs
        print(f"Fourier freqs: {self.freqs}")


class Dataset_ETT_minute_PCA(Dataset_ETT_minute):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., trend_k=0.02, rank_ratio=1.0, 
        input_rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0, load_from_disk="", **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.load_from_disk = load_from_disk
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        self.pca_components = None
        self.input_components = None

        if self.set_type != 0:
            return

        print("Fitting PCA ...")
        if self.load_from_disk:
            proj_dir = os.path.join(self.load_from_disk, f"sp{self.speedup_sklearn}")
            if os.path.exists(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy')):
                self.input_components = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'))
                self.input_initializer = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'))
                self.input_weights = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'))
            if os.path.exists(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy')):
                self.pca_components = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'))
                self.initializer = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'))
                self.weights = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'))

        if self.pca_components is None and self.input_components is None:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                label = label[-self.pred_len:]
                input_seq.append(inp)
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
            self.input_components, self.input_initializer, self.input_weights = get_pca_base(input_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            self.pca_components, self.initializer, self.weights = get_pca_base(label_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            if self.load_from_disk:
                os.makedirs(os.path.join(proj_dir, 'input', f"{self.seq_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'), self.input_components)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'), self.input_initializer)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'), self.input_weights)
                os.makedirs(os.path.join(proj_dir, 'output', f"{self.pred_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'), self.pca_components)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'), self.initializer)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'), self.weights)

        if input_rank_ratio and input_rank_ratio <= 1.0:
            full_rank = self.input_weights.shape[-1]
            proj_dim = int(full_rank * input_rank_ratio)
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]
        elif input_rank_ratio < 0 or input_rank_ratio > 1:
            proj_dim = int(abs(input_rank_ratio))
            proj_dim = min(proj_dim, self.input_weights.shape[-1])
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]

        print(f"Input PCA components shape: {self.input_components.shape}")
        print(f"Input PCA weights shape: {self.input_weights.shape}")

        if rank_ratio and rank_ratio <= 1.0:
            full_rank = self.weights.shape[-1]
            proj_dim = int(full_rank * rank_ratio)
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]
        elif rank_ratio < 0 or rank_ratio > 1:
            proj_dim = int(abs(rank_ratio))
            proj_dim = min(proj_dim, self.weights.shape[-1])
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]

        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")


class Dataset_ETT_minute_CCA(Dataset_ETT_minute):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., trend_k=0.02, rank_ratio=1.0, 
        input_rank_ratio=1.0, pca_dim="all", reinit=0, speedup_sklearn=0, align_type=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.align_type = align_type
        self.speedup_sklearn = speedup_sklearn
        self.cca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def cca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="D", reinit=0):
        if self.set_type != 0:
            self.Wx = None
            self.Wy = None
            return

        print("Fitting CCA ...")
        if self.align_type != 5:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                input_seq.append(inp)
                label = label[-self.pred_len:]
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
        elif self.align_type == 5:
            input_seq = self.data_x[:-self.pred_len]  # shape: [N, D]
            label_seq = self.data_y[self.pred_len:]  # shape: [N, D]
        self.Wx, self.Wy, self.means, self.stds = get_cca_projection(
            input_seq, label_seq, rank_ratio, pca_dim, self.speedup_sklearn, self.align_type
        )
        print(f"CCA Wx shape: {self.Wx.shape}")
        print(f"CCA Wy shape: {self.Wy.shape}")


class Dataset_Custom(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.add_noise = add_noise
        self.noise_amp = noise_amp
        self.noise_freq_percentage = noise_freq_percentage
        self.noise_seed = noise_seed
        self.noise_type = noise_type
        self.data_percentage = data_percentage

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.set_type == 0:
            print('Head lines of raw dataframe:')
            print(df_raw.head(5))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0] if (num_train + num_vali) % 2 == 0 else [1]
        border1s += [num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0 and self.data_percentage < 1.:  # train data
            print(f"Shrink the train data to {self.data_percentage * 100}%")
            border1 = border2 - int((border2 - border1) * self.data_percentage)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.add_noise and self.noise_amp > 0:
            if self.noise_type == 'normal':
                print(f"Add normal noise to the raw data with amplitude: {self.noise_amp}")
                data_len = border2s[1] - border1s[0]
                tmp_data = df_data[border1s[0]:border2s[1]].copy()
                freq_domain = np.fft.rfft(tmp_data, axis=0)
                freq_domain += self.noise_amp
                noise_data = np.fft.irfft(freq_domain, axis=0).real
                df_data[border1s[0]:border2s[1]] = noise_data

            elif self.noise_type == 'sin':
                print(f"Add sin noise to the raw data with amplitude: {self.noise_amp}")
                data_len = border2s[1] - border1s[0]
                noise_freq = int(self.noise_freq_percentage * (data_len // 2 + 1))
                tmp_data = df_data[border1s[0]:border2s[1]].copy()
                freq_domain = np.fft.rfft(tmp_data, axis=0)
                freq_domain[-noise_freq:] += self.noise_amp
                noise_data = np.fft.irfft(freq_domain, axis=0).real
                df_data[border1s[0]:border2s[1]] = noise_data
            else:
                raise NotImplementedError(f"Unrecognized noise type: {self.noise_type}")

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_Fourier(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., num_freqs=16, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.fourier_fit(num_freqs)

    def fourier_fit(self, num_freqs=10):
        if self.set_type != 0:
            self.freqs = None
            return

        print("Fitting fourier ...")
        f = fourier(num_freqs=num_freqs)
        f.fft(self.data_x)
        self.freqs = f.freqs
        print(f"Fourier freqs: {self.freqs}")


class Dataset_Custom_PCA(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, load_from_disk="", **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.load_from_disk = load_from_disk
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        self.pca_components = None
        self.input_components = None

        if self.set_type != 0:
            return

        print("Fitting PCA ...")
        if self.load_from_disk:
            proj_dir = os.path.join(self.load_from_disk, f"sp{self.speedup_sklearn}")
            if os.path.exists(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy')):
                self.input_components = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'))
                self.input_initializer = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'))
                self.input_weights = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'))
            if os.path.exists(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy')):
                self.pca_components = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'))
                self.initializer = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'))
                self.weights = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'))

        if self.pca_components is None and self.input_components is None:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                label = label[-self.pred_len:]
                input_seq.append(inp)
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
            self.input_components, self.input_initializer, self.input_weights = get_pca_base(input_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            self.pca_components, self.initializer, self.weights = get_pca_base(label_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            if self.load_from_disk:
                os.makedirs(os.path.join(proj_dir, 'input', f"{self.seq_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'), self.input_components)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'), self.input_initializer)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'), self.input_weights)
                os.makedirs(os.path.join(proj_dir, 'output', f"{self.pred_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'), self.pca_components)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'), self.initializer)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'), self.weights)

        if input_rank_ratio and input_rank_ratio <= 1.0:
            full_rank = self.input_weights.shape[-1]
            proj_dim = int(full_rank * input_rank_ratio)
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]
        elif input_rank_ratio < 0 or input_rank_ratio > 1:
            proj_dim = int(abs(input_rank_ratio))
            proj_dim = min(proj_dim, self.input_weights.shape[-1])
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]

        print(f"Input PCA components shape: {self.input_components.shape}")
        print(f"Input PCA weights shape: {self.input_weights.shape}")

        if rank_ratio and rank_ratio <= 1.0:
            full_rank = self.weights.shape[-1]
            proj_dim = int(full_rank * rank_ratio)
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]
        elif rank_ratio < 0 or rank_ratio > 1:
            proj_dim = int(abs(rank_ratio))
            proj_dim = min(proj_dim, self.weights.shape[-1])
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]

        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")


class Dataset_Custom_FA(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.fa_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def fa_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        if self.set_type != 0:
            self.pca_components = None
            self.input_components = None
            return

        print("Fitting FA ...")
        input_seq, label_seq = [], []
        for i in range(self.__len__()):
            inp, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            input_seq.append(inp)
            label_seq.append(label)
        input_seq = np.array(input_seq)  # shape: [N, S, D]
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.input_components, self.input_initializer, self.input_mean = get_fa_base(input_seq, input_rank_ratio, pca_dim, reinit)
        self.fa_components, self.initializer, self.fa_mean = get_fa_base(label_seq, rank_ratio, pca_dim, reinit)
        print(f"Input FA components shape: {self.input_components.shape}")
        print(f"Input FA mean shape: {self.input_mean.shape}")
        print(f"FA components shape: {self.fa_components.shape}")
        print(f"FA mean shape: {self.fa_mean.shape}")


class Dataset_Custom_RobustPCA(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        if self.set_type != 0:
            self.pca_components = None
            self.input_components = None
            return

        print("Fitting Robust PCA ...")
        input_seq, label_seq = [], []
        for i in range(self.__len__()):
            inp, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            input_seq.append(inp)
            label_seq.append(label)
        input_seq = np.array(input_seq)  # shape: [N, S, D]
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.input_components, self.input_initializer, self.input_mean = get_robustpca_base(input_seq, input_rank_ratio, pca_dim, reinit)
        self.pca_components, self.initializer, self.rpca_mean = get_robustpca_base(label_seq, rank_ratio, pca_dim, reinit)
        print(f"Input PCA components shape: {self.input_components.shape}")
        print(f"Input PCA mean shape: {self.input_mean.shape}")
        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA mean shape: {self.rpca_mean.shape}")


class Dataset_Custom_SVD(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        if self.set_type != 0:
            self.svd_components = None
            self.input_components = None
            return

        print("Fitting SVD ...")
        input_seq, label_seq = [], []
        for i in range(self.__len__()):
            inp, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            input_seq.append(inp)
            label_seq.append(label)
        input_seq = np.array(input_seq)  # shape: [N, S, D]
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.input_components, self.input_initializer = get_svd_base(input_seq, input_rank_ratio, pca_dim, reinit)
        self.svd_components, self.initializer = get_svd_base(label_seq, rank_ratio, pca_dim, reinit)
        print(f"Input SVD components shape: {self.input_components.shape}")
        print(f"SVD components shape: {self.svd_components.shape}")


class Dataset_Custom_ICA(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.ica_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def ica_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        if self.set_type != 0:
            self.ica_components = None
            self.input_components = None
            return

        print("Fitting ICA ...")
        input_seq, label_seq = [], []
        for i in range(self.__len__()):
            inp, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            input_seq.append(inp)
            label_seq.append(label)
        input_seq = np.array(input_seq)  # shape: [N, S, D]
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.input_components, self.input_initializer, self.input_mean, self.input_whitening = get_ica_base(input_seq, input_rank_ratio, pca_dim, reinit)
        self.ica_components, self.initializer, self.ica_mean, self.whitening = get_ica_base(label_seq, rank_ratio, pca_dim, reinit)
        print(f"Input ICA components shape: {self.input_components.shape}")
        print(f"Input ICA mean shape: {self.input_mean.shape}")
        print(f"Input ICA whitening shape: {self.input_whitening.shape}")
        print(f"ICA components shape: {self.ica_components.shape}")
        print(f"ICA mean shape: {self.ica_mean.shape}")
        print(f"ICA whitening shape: {self.whitening.shape}")


class Dataset_Custom_RobustICA(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.ica_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def ica_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        if self.set_type != 0:
            self.ica_components = None
            self.input_components = None
            return

        print("Fitting Robust ICA ...")
        input_seq, label_seq = [], []
        for i in range(self.__len__()):
            inp, label, _, _ = self.__getitem__(i)
            label = label[-self.pred_len:]
            input_seq.append(inp)
            label_seq.append(label)
        input_seq = np.array(input_seq)  # shape: [N, S, D]
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.input_components, self.input_initializer = get_robustica_base(input_seq, input_rank_ratio, pca_dim, reinit)
        self.ica_components, self.initializer = get_robustica_base(label_seq, rank_ratio, pca_dim, reinit)
        print(f"Input Robust ICA components shape: {self.input_components.shape}")
        print(f"Robust ICA components shape: {self.ica_components.shape}")


class Dataset_Custom_CCA(Dataset_Custom):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='electricity.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, align_type=0,  **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns, 
            add_noise, noise_amp, noise_freq_percentage, noise_seed, noise_type, data_percentage, **kwargs
        )

        self.align_type = align_type
        self.speedup_sklearn = speedup_sklearn
        self.cca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def cca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="D", reinit=0):
        if self.set_type != 0:
            self.Wx = None
            self.Wy = None
            return

        print("Fitting CCA ...")
        if self.align_type != 5:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                input_seq.append(inp)
                label = label[-self.pred_len:]
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
        elif self.align_type == 5:
            input_seq = self.data_x[:-self.pred_len]  # shape: [N, D]
            label_seq = self.data_y[self.pred_len:]  # shape: [N, D]
        self.Wx, self.Wy, self.means, self.stds = get_cca_projection(
            input_seq, label_seq, rank_ratio, pca_dim, self.speedup_sklearn, self.align_type
        )
        print(f"CCA Wx shape: {self.Wx.shape}")
        print(f"CCA Wy shape: {self.Wy.shape}")


class Dataset_Synthetic(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', 
        target='OT', scale=True, timeenc=0, freq='h', **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        if self.set_type == 0:
            print('Head lines of raw dataframe:')
            print(df_raw.head(5))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # data_stamp = df_raw[['time']][border1:border2].values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS_PCA(Dataset_PEMS):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, load_from_disk="", **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.load_from_disk = load_from_disk
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        self.pca_components = None
        self.input_components = None

        if self.set_type != 0:
            return

        print("Fitting PCA ...")
        if self.load_from_disk:
            proj_dir = os.path.join(self.load_from_disk, f"sp{self.speedup_sklearn}")
            if os.path.exists(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy')):
                self.input_components = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'))
                self.input_initializer = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'))
                self.input_weights = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'))
            if os.path.exists(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy')):
                self.pca_components = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'))
                self.initializer = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'))
                self.weights = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'))

        if self.pca_components is None and self.input_components is None:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                label = label[-self.pred_len:]
                input_seq.append(inp)
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
            self.input_components, self.input_initializer, self.input_weights = get_pca_base(input_seq, input_rank_ratio, pca_dim, reinit, self.speedup_sklearn)
            self.pca_components, self.initializer, self.weights = get_pca_base(label_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            if self.load_from_disk:
                os.makedirs(os.path.join(proj_dir, 'input', f"{self.seq_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'), self.input_components)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'), self.input_initializer)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'), self.input_weights)
                os.makedirs(os.path.join(proj_dir, 'output', f"{self.pred_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'), self.pca_components)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'), self.initializer)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'), self.weights)

        if input_rank_ratio and input_rank_ratio <= 1.0:
            full_rank = self.input_weights.shape[-1]
            proj_dim = int(full_rank * input_rank_ratio)
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]
        elif input_rank_ratio < 0 or input_rank_ratio > 1:
            proj_dim = int(abs(input_rank_ratio))
            proj_dim = min(proj_dim, self.input_weights.shape[-1])
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]

        print(f"Input PCA components shape: {self.input_components.shape}")
        print(f"Input PCA weights shape: {self.input_weights.shape}")

        if rank_ratio and rank_ratio <= 1.0:
            full_rank = self.weights.shape[-1]
            proj_dim = int(full_rank * rank_ratio)
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]
        elif rank_ratio < 0 or rank_ratio > 1:
            proj_dim = int(abs(rank_ratio))
            proj_dim = min(proj_dim, self.weights.shape[-1])
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]

        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")


class Dataset_PEMS_CCA(Dataset_PEMS):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, align_type=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, timeenc, freq, **kwargs
        )

        self.align_type = align_type
        self.speedup_sklearn = speedup_sklearn
        self.cca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def cca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="D", reinit=0):
        if self.set_type != 0:
            self.Wx = None
            self.Wy = None
            return

        print("Fitting CCA ...")
        if self.align_type != 5:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                input_seq.append(inp)
                label = label[-self.pred_len:]
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
        elif self.align_type == 5:
            input_seq = self.data_x[:-self.pred_len]  # shape: [N, D]
            label_seq = self.data_y[self.pred_len:]  # shape: [N, D]
        self.Wx, self.Wy, self.means, self.stds = get_cca_projection(
            input_seq, label_seq, rank_ratio, pca_dim, self.speedup_sklearn, self.align_type
        )
        print(f"CCA Wx shape: {self.Wx.shape}")
        print(f"CCA Wy shape: {self.Wy.shape}")


class Dataset_Solar(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='ETTh1.csv', 
        target='OT', scale=True, timeenc=0, freq='h', **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(
        self, root_path, flag='pred', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, inverse=False, timeenc=0, freq='15min', seasonal_patterns='Yearly', 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]  # 2 * pred_len
        self.label_len = size[1]
        self.pred_len = size[2]  # corresponds to seasonal pattern

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = [
            np.array(v[~np.isnan(v)]) 
            for v in dataset.values[dataset.groups == self.seasonal_patterns]
        ]
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]  # not fixed length
        cut_point = np.random.randint(
            low=max(1, len(sampled_timeseries) - self.window_sampling_limit),  # pred_len * history_size
            high=len(sampled_timeseries),
            size=1
        )[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
            cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)
        ]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_M4_PCA(Dataset_M4):
    def __init__(
        self, root_path, flag='pred', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, inverse=False, timeenc=0, freq='15min', seasonal_patterns='Yearly', 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, load_from_disk="", **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, inverse,
            timeenc, freq, seasonal_patterns, add_noise, noise_amp,
            noise_freq_percentage, noise_seed, noise_type,
            data_percentage, **kwargs
        )

        self.speedup_sklearn = speedup_sklearn
        self.load_from_disk = load_from_disk
        self.pca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def pca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", reinit=0):
        self.pca_components = None
        self.input_components = None

        if self.flag != 'train':
            return

        print("Fitting PCA ...")
        if self.load_from_disk:
            proj_dir = os.path.join(self.load_from_disk, f"sp{self.speedup_sklearn}")
            if os.path.exists(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy')):
                self.input_components = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'))
                self.input_initializer = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'))
                self.input_weights = np.load(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'))
            if os.path.exists(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy')):
                self.pca_components = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'))
                self.initializer = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'))
                self.weights = np.load(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'))

        if self.pca_components is None and self.input_components is None:
            input_seq, label_seq = [], []
            for i in range(self.__len__()):
                inp, label, _, _ = self.__getitem__(i)
                label = label[-self.pred_len:]
                input_seq.append(inp)
                label_seq.append(label)
            input_seq = np.array(input_seq)  # shape: [N, S, D]
            label_seq = np.array(label_seq)  # shape: [N, P, D]
            self.input_components, self.input_initializer, self.input_weights = get_pca_base(input_seq, input_rank_ratio, pca_dim, reinit, self.speedup_sklearn)
            self.pca_components, self.initializer, self.weights = get_pca_base(label_seq, 1.0, pca_dim, reinit, self.speedup_sklearn)
            if self.load_from_disk:
                os.makedirs(os.path.join(proj_dir, 'input', f"{self.seq_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'pca_components.npy'), self.input_components)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'initializer.npy'), self.input_initializer)
                np.save(os.path.join(proj_dir, 'input', f"{self.seq_len}", 'weights.npy'), self.input_weights)
                os.makedirs(os.path.join(proj_dir, 'output', f"{self.pred_len}"), exist_ok=True)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'pca_components.npy'), self.pca_components)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'initializer.npy'), self.initializer)
                np.save(os.path.join(proj_dir, 'output', f"{self.pred_len}", 'weights.npy'), self.weights)

        if input_rank_ratio and input_rank_ratio <= 1.0:
            full_rank = self.input_weights.shape[-1]
            proj_dim = int(full_rank * input_rank_ratio)
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]
        elif input_rank_ratio < 0 or input_rank_ratio > 1:
            proj_dim = int(abs(input_rank_ratio))
            proj_dim = min(proj_dim, self.input_weights.shape[-1])
            self.input_components = self.input_components[:proj_dim] if pca_dim == "all" else self.input_components[:, :proj_dim]
            self.input_weights = self.input_weights[:proj_dim] if pca_dim == "all" else self.input_weights[:, :proj_dim]

        print(f"Input PCA components shape: {self.input_components.shape}")
        print(f"Input PCA weights shape: {self.input_weights.shape}")

        if rank_ratio and rank_ratio <= 1.0:
            full_rank = self.weights.shape[-1]
            proj_dim = int(full_rank * rank_ratio)
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]
        elif rank_ratio < 0 or rank_ratio > 1:
            proj_dim = int(abs(rank_ratio))
            proj_dim = min(proj_dim, self.weights.shape[-1])
            self.pca_components = self.pca_components[:proj_dim] if pca_dim == "all" else self.pca_components[:, :proj_dim]
            self.weights = self.weights[:proj_dim] if pca_dim == "all" else self.weights[:, :proj_dim]

        print(f"PCA components shape: {self.pca_components.shape}")
        print(f"PCA weights shape: {self.weights.shape}")


class Dataset_M4_CCA(Dataset_M4):
    def __init__(
        self, root_path, flag='pred', size=None, features='S', data_path='ETTh1.csv',
        target='OT', scale=True, inverse=False, timeenc=0, freq='15min', seasonal_patterns='Yearly', 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="all", 
        reinit=0, speedup_sklearn=0, align_type=0, **kwargs
    ):
        super().__init__(
            root_path, flag, size, features, data_path, target, scale, inverse,
            timeenc, freq, seasonal_patterns, add_noise, noise_amp,
            noise_freq_percentage, noise_seed, noise_type,
            data_percentage, **kwargs
        )

        self.align_type = align_type
        self.speedup_sklearn = speedup_sklearn
        self.cca_fit(rank_ratio, input_rank_ratio, pca_dim, reinit)

    def cca_fit(self, rank_ratio=1.0, input_rank_ratio=1.0, pca_dim="D", reinit=0):
        if self.set_type != 0:
            self.Wx = None
            self.Wy = None
            return

        print("Fitting CCA ...")
        assert self.align_type != 5, "M4 dataset does not support align_type=5"
        input_seq, label_seq = [], []
        for i in range(self.__len__()):
            inp, label, _, _ = self.__getitem__(i)
            input_seq.append(inp)
            label = label[-self.pred_len:]
            label_seq.append(label)
        input_seq = np.array(input_seq)  # shape: [N, S, D]
        label_seq = np.array(label_seq)  # shape: [N, P, D]
        self.Wx, self.Wy, self.means, self.stds = get_cca_projection(
            input_seq, label_seq, rank_ratio, pca_dim, self.speedup_sklearn, self.align_type
        )
        print(f"CCA Wx shape: {self.Wx.shape}")
        print(f"CCA Wy shape: {self.Wy.shape}")


class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)


class Dataset_SRU(Dataset):
    def __init__(
        self, root_path, flag='train', size=None, features='S', data_path='SRU_data.txt',
        target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None, 
        add_noise=False, noise_amp=0.1, noise_freq_percentage=0.05, noise_seed=2023, 
        noise_type='sin', data_percentage=1., shift=0, **kwargs
    ):
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.shift = shift

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def shift_data(self, data_raw, columns):
        if self.shift == 0 or len(columns) == 0:
            return data_raw, []

        data = data_raw.copy()
        for col in columns:
            for i in range(1, self.shift + 1):
                data[f'{col}_{i}'] = data[col].shift(i)
        shifted_columns = [f'{col}_{i}' for col in columns for i in range(1, self.shift + 1)]
        data = data.iloc[self.shift:].reset_index(drop=True)
        return data, shifted_columns

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(
            os.path.join(self.root_path, self.data_path), header=None, skiprows=1, skip_blank_lines=True, 
            sep='  ', engine='python', names=['u1', 'u2', 'u3', 'u4', 'u5', 'y1', 'y2']
        )

        df_raw, shifted_columns = self.shift_data(df_raw, ['y1', 'y2'])
        df_raw = df_raw[['u1', 'u2', 'u3', 'u4', 'u5'] + shifted_columns + ['y1', 'y2']]

        if self.set_type == 0:
            print('Head lines of raw dataframe:')
            print(df_raw.head(5))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0] if (num_train + num_vali) % 2 == 0 else [1]
        border1s += [num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            df_data = df_raw
        elif self.features == 'S':
            df_data = df_raw.iloc[:, -2:]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
