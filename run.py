import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import os
import random
import torch

import cupy as cp
import numpy as np

from exp import EXP_DICT
from utils.print_args import print_args
from utils.tools import EvalAction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--fix_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--rerun', type=int, help='rerun', default=0)
    parser.add_argument('--verbose', type=int, help='verbose', default=0)
    parser.add_argument('--use_profiler', type=int, help='use profiler', default=0)

    # save
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--results', type=str, default='./results/', help='location of results')
    parser.add_argument('--test_results', type=str, default='./test_results/', help='location of test results')
    parser.add_argument('--log_path', type=str, default='./result_long_term_forecast.txt', help='log path')
    parser.add_argument('--log_step', type=int, default=10, help='log step')
    parser.add_argument('--output_pred', action='store_true', help='output true and pred', default=False)
    parser.add_argument('--output_vis', action='store_true', help='output visual figures', default=False)
    parser.add_argument('--output_log', action='store_true', help='output log', default=False)
    parser.add_argument('--report_to', type=str, default='local', help='report to tensorboard or None')

    # data loader
    parser.add_argument('--data_id', type=str, default='ETTm1', help='dataset name')
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options: [M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--add_noise', action='store_true', help='add noise')
    parser.add_argument('--noise_amp', type=float, default=1, help='noise ampitude')
    parser.add_argument('--noise_freq_percentage', type=float, default=0.05, help='noise frequency percentage')
    parser.add_argument('--noise_seed', type=int, default=2023, help='noise seed')
    parser.add_argument('--noise_type', type=str, default='sin', help='noise type, options: [sin, normal]')
    parser.add_argument('--cutoff_freq_percentage', type=float, default=0.06, help='cutoff frequency')
    parser.add_argument('--data_percentage', type=float, default=1., help='percentage of training data')
    parser.add_argument('--shift', type=int, default=0, help='shift of time series')
    parser.add_argument('--num_freqs', type=int, default=16, help='number of frequencies')
    parser.add_argument('--speedup_sklearn', type=int, default=0, help='1: use sklearnex, 0: use sklearn, 2: cuML')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--reconstruction_type', type=str, default="imputation", help='type of reconstruction')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options: [timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='1: channel dependence 0: channel independence for FreTS model')
    parser.add_argument('--scales', default=[16, 8, 4, 2, 1], help='scales in mult-scale')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for upsample')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')

    # optimization
    parser.add_argument('--optim_type', type=str, default='adam', help='optimizer type')
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--auxi_batch_size', type=int, default=1024, help='batch size of test input data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size of test input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', action=EvalAction, default='type1', help='adjust learning rate')
    parser.add_argument('--step_size', type=int, default=1, help='step size for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--mode', type=int, default=0, help='mode for learning rate decay')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--pct_start', type=float, default=0.2, help='Warmup ratio for the learning rate scheduler')

    # FreDF
    parser.add_argument('--rec_lambda', type=float, default=0., help='weight of reconstruction function')
    parser.add_argument('--auxi_lambda', type=float, default=1, help='weight of auxilary function')
    parser.add_argument('--auxi_loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--auxi_mode', type=str, default='fft', help='auxi loss mode, options: [fft, rfft]')
    parser.add_argument('--auxi_type', type=str, default='complex', help='auxi loss type, options: [complex, mag, phase, mag-phase]')
    parser.add_argument('--module_first', type=int, default=1, help='calculate module first then mean ')
    parser.add_argument('--leg_degree', type=int, default=2, help='degree of legendre polynomial')
    parser.add_argument('--alpha', type=float, default=0.5, help="weight for dilate loss")
    parser.add_argument('--dilate_alpha', type=float, default=0.5, help="weight for dilate loss")
    parser.add_argument('--gamma', type=float, default=0.01, help="coef for dilate loss")

    # PCA
    parser.add_argument('--rank_ratio', type=float, default=1.0, help='ratio of low rank for PCA')
    parser.add_argument('--input_rank_ratio', type=float, default=1.0, help='ratio of low rank for input PCA')
    parser.add_argument('--input_trans', type=str, default='None', help='input transform, options: [basis, fourier]')
    parser.add_argument('--pca_dim', type=str, default="all", help="dimension for PCA, choices in ['all','T','D']")
    parser.add_argument('--reinit', type=int, default=0, help="whether reinit for PCA")
    parser.add_argument('--input_reinit', type=int, default=0, help="whether reinit for input PCA")
    parser.add_argument('--dist_scale', type=float, default=0.1, help="scale factor for ot distance matrix")
    parser.add_argument('--use_weights', type=int, default=0, help="use pca weights or not")
    parser.add_argument('--input_use_weights', type=int, default=0, help="use input pca weights or not")
    parser.add_argument('--load_from_disk', type=str, default="")

    # CCA
    parser.add_argument('--align_type', type=int, default=0, help='alignment type; 0: mean')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--thread', type=int, default=1)
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # FBM
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--kernel_size', help='decomposition-kernel', action=EvalAction, default=24)
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--patch_num', type=int, default=14, help='patch number')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

    # ASHyper
    parser.add_argument('--window_size', action=EvalAction, default=[4, 4])
    parser.add_argument('--CSCM', type=str, default='Bottleneck_Construct')
    parser.add_argument('-inner_size', type=int, default=5)
    parser.add_argument('--hyper_num', action=EvalAction, default=[50, 20, 10])
    parser.add_argument('--k', type=int, default=3)

    # FFT OT
    parser.add_argument("--pretrain_model_path", default=None, type=str)
    parser.add_argument('--joint_forecast', type=int, default=0, help='joint forecast; True 1 False 0')
    parser.add_argument('--ot_type', type=str, default='emd1d_h', help="type of ot distance, choices in ['emd1d_h']")
    parser.add_argument('--distance', type=str, default="time", help="distance metric for ot")
    parser.add_argument('--normalize', type=int, default=1, help="normalize ot distance matrix")
    parser.add_argument('--reg_sk', type=float, default=0.1, help="strength of entropy regularization in Sinkhorn")
    parser.add_argument('--numItermax', type=int, default=10000, help="max number of iterations in Sinkhorn")
    parser.add_argument('--stopThr', type=float, default=1e-4, help="stop threshold in Sinkhorn")
    parser.add_argument('--mask_factor', type=float, default=0.01, help="mask factor for mask matrix")

    # SimpleTM
    parser.add_argument('--l1_weight', type=float, default=5e-5, help='Weight of L1 loss')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--wv', type=str, default='db1', help='Wavelet filter type. Supports all wavelets available in PyTorch Wavelets')
    parser.add_argument('--m', type=int, default=3, help='Number of levels for the stationary wavelet transform')
    parser.add_argument('--geomattn_dropout', type=float, default=0.5, help='dropout rate of the projection layer in the geometric attention')
    parser.add_argument('--requires_grad', type=bool, default=True, help='Set to True to enable learnable wavelets')

    # Fredformer
    parser.add_argument('--cf_dim',         type=int, default=48)   #feature dimension
    parser.add_argument('--cf_drop',        type=float, default=0.2)#dropout
    parser.add_argument('--cf_depth',       type=int, default=2)    #Transformer layer
    parser.add_argument('--cf_heads',       type=int, default=6)    #number of multi-heads
    #parser.add_argument('--cf_patch_len',  type=int, default=16)   #patch length
    parser.add_argument('--cf_mlp',         type=int, default=128)  #ff dimension
    parser.add_argument('--cf_head_dim',    type=int, default=32)   #dimension for single head
    parser.add_argument('--cf_weight_decay',type=float, default=0)  #weight_decay
    parser.add_argument('--cf_p',           type=int, default=1)    #patch_type
    parser.add_argument('--use_nys',           type=int, default=0)    #use nystrom
    parser.add_argument('--mlp_drop',           type=float, default=0.3)    #output type
    parser.add_argument('--ablation',       type=int, default=0)    #ablation study 012.
    parser.add_argument('--mlp_hidden', type=int, default=64, help='hidden layer dimension of model')

    # TimeKAN
    parser.add_argument('--begin_order', type=int, default=1, help='begin_order')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='whether to use future_temporal_feature; True 1 False 0')

    # Meta
    parser.add_argument('--meta_lr', type=float, default=0.0005, help='meta learning rate')
    parser.add_argument('--inner_lr', type=float, default=0.0005, help='inner learning rate')
    parser.add_argument('--meta_inner_steps', type=int, default=1, help='meta inner steps')
    parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks')
    parser.add_argument('--overlap_ratio', type=float, default=0.15, help='overlap ratio between tasks')
    parser.add_argument('--meta_optim_type', type=str, default='sgd', help='optimizer type')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max norm for gradient clipping')
    parser.add_argument('--first_order', type=int, default=1, help='first order approximation; True 1 False 0')
    parser.add_argument('--model_per_task', type=int, default=0, help='separate model for each task; True 1 False 0')
    parser.add_argument('--meta_type', type=str, default='all', help='meta learning type')
    parser.add_argument('--weighting_type', type=str, default='softmax', help='type of weighting for auxi loss, options: [softmax, minmax]')
    parser.add_argument('--hyper_dim', type=int, default=2048, help='dimension of hypernet')
    parser.add_argument('--sample_attn', type=int, default=0, help='sample attention; True 1 False 0')

    parser.add_argument('--extra_metrics', action=EvalAction, default=[], help='extra_metrics')
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--warping_length', type=int, default=96, help='warping_length')
    parser.add_argument('--bandwidth', type=float, default=0.)

    args = parser.parse_args()

    fix_seed = args.fix_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    cp.random.seed(fix_seed)

    torch.set_num_threads(args.thread)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name in EXP_DICT:
        Exp = EXP_DICT[args.task_name]

    # setproctitle.setproctitle(args.task_name)

    if args.speedup_sklearn == 1:
        from sklearnex import patch_sklearn
        patch_sklearn()

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.auxi_lambda,
                args.rec_lambda,
                args.auxi_loss,
                args.module_first,
                args.des,
                ii
            )

            if not args.rerun and os.path.exists(os.path.join(args.results, setting, "metrics.npy")):
                print(f">>>>>>>setting {setting} already run, skip")
                sys.exit(0)

            exp = Exp(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.use_profiler:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(args.results, setting)),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
                ) as prof:
                    exp.train(setting, prof=prof)
            else:
                exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()

    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_ax{}_rl{}_axl{}_mf{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.auxi_lambda,
            args.rec_lambda,
            args.auxi_loss,
            args.module_first,
            args.des,
            ii
        )

        exp = Exp(args)  # set experiments

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
