#!/bin/bash
MAX_JOBS=64
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
GPUS=($(seq 0 $((GPU_COUNT - 1))))
TOTAL_GPUS=$GPU_COUNT

get_gpu_allocation(){
    local job_number=$1
    # Calculate which GPU to allocate based on the job number
    local gpu_id=${GPUS[$((job_number % TOTAL_GPUS))]}
    echo $gpu_id
}

check_jobs(){
    while true; do
        jobs_count=$(jobs -p | wc -l)
        if [ "$jobs_count" -lt "$MAX_JOBS" ]; then
            break
        fi
        sleep 1
    done
}

job_number=0

DATA_ROOT=$USRDIR/dataset
OUT_ROOT=/mnt/tidalfs-bdsz01/dataset/llm_ckpt/plc_data/Time-o1
EXP_NAME=long_term_Tucker
seed=2023
des='Fredformer'

model_name=Fredformer

auxi_mode=basis
auxi_type=pca
pca_dim=Tucker
test_batch_size=1

# datasets to run
datasets=(ETTh2)



# hyper-parameters
dst=ETTh1

pl_list=(96 192 336 720)
lbd_list=(0.0 0.1 0.2 0.3)
lr_list=(0.0005 0.001 0.002 0.005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(1)
lradj_list=(type3 type1)
bs_list=(128)
use_weights_list=(0)


lradj=type3
train_epochs=100
patience=10
batch_size=128
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh1_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --patch_len 4 \
            --stride 4 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim\
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=ETTh2

# pl_list=(96 192 336 720)
# lbd_list=(0.0 0.1 0.2 0.4 0.6)
# lr_list=(0.0005 0.001 0.002 0.005)
# rank_ratio_T_list=(1.0 0.9 0.8 0.7 0.6)
# rank_ratio_D_list=(1.0 0.9 0.8)
# reinit_list=(1)
# auxi_loss_list=(MAE)
# out_chan_indep_list=(1)
# lradj_list=(type3 type1)
# bs_list=(128)
# use_weights_list=(0)

pl_list=(96 336)
lbd_list=(0.0 0.1 0.2 0.3 0.8 0.9)
lr_list=(0.003 0.004 0.002 0.005 0.01)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(1)
lradj_list=(type3 type1 TST)
bs_list=(128)
use_weights_list=(0)


lradj=type3
train_epochs=100
patience=10
batch_size=128
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        192) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTh2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTh2_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 4 \
            --d_model $d_model \
            --d_ff 128 \
            --dropout 0.3 \
            --fc_dropout 0.3 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done





# hyper-parameters
dst=ETTm1

pl_list=(96 192 336 720)
lbd_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.8)
lr_list=(0.0005 0.001 0.002 0.005)
rank_ratio_T_list=(1.0 0.9)
rank_ratio_D_list=(1.0 0.9)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(1)
lradj_list=(TST)
bs_list=(128)
use_weights_list=(0)

lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        192) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm1.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm1_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.4 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done







# hyper-parameters
dst=ETTm2

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4 0.6)
lr_list=(0.0005 0.001 0.002 0.005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(1)
lradj_list=(TST)
bs_list=(128)
use_weights_list=(0)

lradj=TST
train_epochs=100
patience=10
batch_size=128
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        192) cf_dim=164 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=48;;
        336) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
        720) cf_dim=128 cf_depth=2 cf_heads=8 cf_mlp=96 cf_head_dim=32 d_model=24;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/ETT-small/ \
            --data_path ETTm2.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data ETTm2_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 4 \
            --stride 4 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.4 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=ECL

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.01 0.005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(0 1)
lradj_list=(type3)
bs_list=(32)
use_weights_list=(0)

lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        192) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        336) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        720) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/electricity/ \
            --data_path electricity.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done





# hyper-parameters
dst=Traffic

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.01 0.005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(0 1)
lradj_list=(type3)
bs_list=(32)
use_weights_list=(0)

lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=512 cf_depth=3 cf_heads=8 cf_mlp=512 cf_head_dim=32 d_model=256;;
        192) cf_dim=512 cf_depth=3 cf_heads=8 cf_mlp=512 cf_head_dim=32 d_model=256;;
        336) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=48 d_model=256;;
        720) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=48 d_model=256;;
    esac
    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/traffic/ \
            --data_path traffic.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done






# hyper-parameters
dst=Weather

pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.001 0.0005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(0 1)
lradj_list=(type3)
bs_list=(128)
use_weights_list=(0)

lradj=type3
train_epochs=100
patience=5
batch_size=128
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        96)  cf_dim=32 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=8;;
        192) cf_dim=32 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=8;;
        336) cf_dim=16 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=4;;
        720) cf_dim=16 cf_depth=3 cf_heads=8 cf_mlp=32 cf_head_dim=8 d_model=4;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/weather/ \
            --data_path weather.csv \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data custom_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 16 \
            --stride 16 \
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done







# hyper-parameters
dst=PEMS03

pl_list=(12 24 36 48)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.01 0.005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(0 1)
lradj_list=(type3)
bs_list=(32)
use_weights_list=(0)

lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12)  cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        24) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        36) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        48) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS03.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done








# hyper-parameters
dst=PEMS08

pl_list=(12 24 36 48)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.01 0.005)
rank_ratio_T_list=(1.0 0.9 0.8)
rank_ratio_D_list=(1.0 0.9 0.8)
reinit_list=(1)
auxi_loss_list=(MAE)
out_chan_indep_list=(0 1)
lradj_list=(type3)
bs_list=(32)
use_weights_list=(0)

lradj=TST
train_epochs=100
patience=5
batch_size=32
test_batch_size=1

rerun=0

for use_weights in ${use_weights_list[@]}; do
for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for out_chan_indep in ${out_chan_indep_list[@]}; do
for rank_ratio_T in ${rank_ratio_T_list[@]}; do
for rank_ratio_D in ${rank_ratio_D_list[@]}; do
for reinit in ${reinit_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for lradj in ${lradj_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    case $pl in
        12)  cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        24) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        36) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
        48) cf_dim=640 cf_depth=3 cf_heads=8 cf_mlp=640 cf_head_dim=32 d_model=256;;
    esac

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    rank_ratio="[${rank_ratio_T},${rank_ratio_D}]"

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${cf_dim}_${cf_depth}_${cf_heads}_${cf_mlp}_${cf_head_dim}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${auxi_mode}_${auxi_type}_${out_chan_indep}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    mkdir -p "${PROJ_DIR}/"

    CHECKPOINTS=$OUTPUT_DIR/checkpoints/
    RESULTS=$OUTPUT_DIR/results/
    TEST_RESULTS=$OUTPUT_DIR/test_results/
    LOG_PATH=$OUTPUT_DIR/result_long_term_forecast.txt

    mkdir -p "${OUTPUT_DIR}/"
    # if rerun, remove the previous stdout
    if [ $rerun -eq 1 ]; then
        rm -rf "${OUTPUT_DIR}/stdout.log"
    else
        subdirs=("$RESULTS"/*)
        if [ ${#subdirs[@]} -eq 1 ] && [ -f "${subdirs[0]}/metrics.yaml" ]; then
            echo ">>>>>>> Job: $JOB_NAME already run, skip <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
            continue
        fi
    fi

    check_jobs
    # Get GPU allocation for this job
    gpu_allocation=$(get_gpu_allocation $job_number)
    # Increment job number for the next iteration
    ((job_number++))

    echo "Running command for $JOB_NAME"
    {
        # Set CUDA_VISIBLE_DEVICES for this script and run it in the background
        CUDA_VISIBLE_DEVICES=$gpu_allocation python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path $DATA_ROOT/PEMS/ \
            --data_path PEMS08.npz \
            --model_id "${dst}_96_${pl}" \
            --model ${model_name} \
            --data_id $dst \
            --data PEMS_PCA \
            --features M \
            --seq_len 96 \
            --label_len 48 \
            --pred_len ${pl} \
            --e_layers 3 \
            --n_heads 16 \
            --d_model $d_model \
            --d_ff 256 \
            --dropout 0.2 \
            --fc_dropout 0.2 \
            --head_dropout 0 \
            --patch_len 48 \
            --stride 48 \
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --des ${des} \
            --learning_rate ${lr} \
            --lradj ${lradj} \
            --pct_start 0.2 \
            --train_epochs ${train_epochs} \
            --patience ${patience} \
            --batch_size ${batch_size} \
            --test_batch_size ${test_batch_size} \
            --itr 1 \
            --rec_lambda ${rl} \
            --auxi_lambda ${ax} \
            --auxi_mode ${auxi_mode} \
            --auxi_type ${auxi_type} \
            --pca_dim ${pca_dim} \
            --rank_ratio ${rank_ratio} \
            --reinit ${reinit} \
            --use_weights ${use_weights} \
            --auxi_loss ${auxi_loss} \
            --fix_seed ${seed} \
            --cf_dim $cf_dim \
            --cf_depth $cf_depth \
            --cf_heads $cf_heads \
            --cf_mlp $cf_mlp \
            --cf_head_dim $cf_head_dim \
            --use_nys 1 \
            --individual 0 \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --out_chan_indep ${out_chan_indep} \
            --speedup_sklearn 2

        sleep 5
    # } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
    } &
done
done
done
done
done
done
done
done
done
done
done






wait