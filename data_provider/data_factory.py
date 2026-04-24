import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_Custom_CCA,
    Dataset_Custom_ICA,
    Dataset_Custom_RobustICA,
    Dataset_Custom_Fourier,
    Dataset_Custom_FA,
    Dataset_Custom_PCA,
    Dataset_Custom_RobustPCA,
    Dataset_Custom_SVD,
    Dataset_ETT_hour,
    Dataset_ETT_hour_CCA,
    Dataset_ETT_hour_Fourier,
    Dataset_ETT_hour_PCA,
    Dataset_ETT_hour_Trend,
    Dataset_ETT_minute,
    Dataset_ETT_minute_CCA,
    Dataset_ETT_minute_Fourier,
    Dataset_ETT_minute_PCA,
    Dataset_M4,
    Dataset_M4_CCA,
    Dataset_M4_PCA,
    Dataset_PEMS,
    Dataset_PEMS_CCA,
    Dataset_PEMS_PCA,
    Dataset_Solar,
    Dataset_SRU,
    MSLSegLoader,
    PSMSegLoader,
    SMAPSegLoader,
    SMDSegLoader,
    SWATSegLoader,
    UEAloader,
)
from data_provider.uea import collate_fn


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh1_Fourier': Dataset_ETT_hour_Fourier,
    'ETTh1_Trend': Dataset_ETT_hour_Trend,
    'ETTh1_PCA': Dataset_ETT_hour_PCA,
    'ETTh1_CCA': Dataset_ETT_hour_CCA,
    'ETTh2': Dataset_ETT_hour,
    'ETTh2_PCA': Dataset_ETT_hour_PCA,
    'ETTh2_CCA': Dataset_ETT_hour_CCA,
    'ETTm1': Dataset_ETT_minute,
    'ETTm1_Fourier': Dataset_ETT_minute_Fourier,
    'ETTm1_PCA': Dataset_ETT_minute_PCA,
    'ETTm1_CCA': Dataset_ETT_minute_CCA,
    'ETTm2': Dataset_ETT_minute,
    'ETTm2_PCA': Dataset_ETT_minute_PCA,
    'ETTm2_CCA': Dataset_ETT_minute_CCA,
    'custom': Dataset_Custom,
    'custom_Fourier': Dataset_Custom_Fourier,
    'custom_FA': Dataset_Custom_FA,
    'custom_PCA': Dataset_Custom_PCA,
    'custom_RobustPCA': Dataset_Custom_RobustPCA,
    'custom_SVD': Dataset_Custom_SVD,
    'custom_ICA': Dataset_Custom_ICA,
    'custom_RobustICA': Dataset_Custom_RobustICA,
    'custom_CCA': Dataset_Custom_CCA,
    'PEMS': Dataset_PEMS,
    'PEMS_PCA': Dataset_PEMS_PCA,
    'PEMS_CCA': Dataset_PEMS_CCA,
    'Solar': Dataset_Solar,
    'm4': Dataset_M4,
    'm4_PCA': Dataset_M4_PCA,
    'm4_CCA': Dataset_M4_CCA,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'SRU': Dataset_SRU,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.test_batch_size  # bsz for test
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns,
            add_noise=args.add_noise,
            noise_amp=args.noise_amp,
            noise_freq_percentage=args.noise_freq_percentage,
            noise_seed=args.noise_seed,
            noise_type=args.noise_type,
            data_percentage=args.data_percentage,
            rank_ratio=args.rank_ratio,
            input_rank_ratio=args.input_rank_ratio,
            pca_dim=args.pca_dim,
            reinit=args.reinit,
            shift=args.shift,
            num_freqs=args.num_freqs,
            speedup_sklearn=args.speedup_sklearn,
            align_type=args.align_type,
            load_from_disk=args.load_from_disk
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
