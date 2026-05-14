#!/bin/bash
MAX_JOBS=48
GPUS=(0 1 2 3 4 5 6 7)
TOTAL_GPUS=${#GPUS[@]}

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
EXP_NAME=long_term
seed=2023
des='FreTS'

model_name=FreTS

input_use_weights=0
auxi_mode=basis
auxi_type=pca
pca_dim=T
use_weights=0
test_batch_size=1

# datasets to run
datasets=(Weather)



# hyper-parameters
dst=ETTh1
pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.001 0.0005 0.0002)
rank_ratio_list=(0.2 0.6 0.8 1.0)
reinit_list=(1)
auxi_loss_list=(MAE)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lradj=type1
train_epochs=10
patience=3
batch_size=32

rerun=0

for auxi_loss in ${auxi_loss_list[@]}; do
for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for reinit in ${reinit_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${input_trans}_${chan_indep}_${extra_rev_in}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
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
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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
pl_list=(96 192 336 720)
lbd_list=(0.0 0.2 0.4)
lr_list=(0.001 0.0005 0.0002)
rank_ratio_list=(0.2 0.6 0.8 1.0)
reinit_list=(1)
auxi_loss_list=(MAE)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lradj=type1
train_epochs=10
patience=3
batch_size=32

rerun=0

for auxi_loss in ${auxi_loss_list[@]}; do
for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for reinit in ${reinit_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${input_trans}_${chan_indep}_${extra_rev_in}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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
lbd_list=(0.0 0.2 0.4)
lr_list=(0.001 0.0005 0.0002)
rank_ratio_list=(0.2 0.6 0.8 1.0)
reinit_list=(1)
auxi_loss_list=(MAE)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lradj=type1
train_epochs=10
patience=3
batch_size=32

rerun=0

for auxi_loss in ${auxi_loss_list[@]}; do
for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for reinit in ${reinit_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${input_trans}_${chan_indep}_${extra_rev_in}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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
lbd_list=(0.0 0.2 0.4)
lr_list=(0.001 0.0005 0.0002)
rank_ratio_list=(0.2 0.6 0.8 1.0)
reinit_list=(1)
auxi_loss_list=(MAE)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lradj=type1
train_epochs=10
patience=3
batch_size=32

rerun=0

for auxi_loss in ${auxi_loss_list[@]}; do
for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for reinit in ${reinit_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${input_trans}_${chan_indep}_${extra_rev_in}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
            --d_model 128 \
            --d_ff 128 \
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
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lambda=1.0

lr=0.0005
lradj=type1
train_epochs=10
patience=3
batch_size=16

rerun=0

for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${input_trans}_${chan_indep}_${extra_rev_in}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 321 \
            --dec_in 321 \
            --c_out 321 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done





# hyper-parameters
dst=Traffic
pl_list=(96 192 336 720)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lambda=1.0

lr=0.0005
lradj=type1
train_epochs=10
patience=3
batch_size=8

rerun=0

for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${input_trans}_${chan_indep}_${extra_rev_in}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 862 \
            --dec_in 862 \
            --c_out 862 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done






# hyper-parameters
dst=Weather

pl_list=(96 192 336 720)
lbd_list=(0.1 0.2 0.5 0.6)
lr_list=(0.0005 0.001)
rank_ratio_list=(0.1 0.3 1.0)
input_reinit_list=(0 1)
input_rank_ratio_list=(0.4 0.6 0.8 1.0)
reinit_list=(1)
auxi_loss_list=(MAE)
extra_rev_in_list=(1 0)
input_trans_list=(evd)
bs_list=(32)

lradj=type3
train_epochs=10
patience=10
batch_size=32

rerun=0

for batch_size in ${bs_list[@]}; do
for auxi_loss in ${auxi_loss_list[@]}; do
for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for reinit in ${reinit_list[@]}; do
for rank_ratio in ${rank_ratio_list[@]}; do
for input_reinit in ${input_reinit_list[@]}; do
for input_rank_ratio in ${input_rank_ratio_list[@]}; do
for lr in ${lr_list[@]}; do
for lambda in ${lbd_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}_${auxi_loss}_${use_weights}_${reinit}_${pca_dim}_${rank_ratio}_${input_trans}_${chan_indep}_${extra_rev_in}_${input_reinit}_${input_rank_ratio}_${auxi_mode}_${auxi_type}_${input_use_weights}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 21 \
            --dec_in 21 \
            --c_out 21 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
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
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_reinit ${input_reinit} \
            --input_rank_ratio ${input_rank_ratio} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --input_use_weights ${input_use_weights} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
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
done






# hyper-parameters
dst=PEMS03
pl_list=(12 24 36 48)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lambda=1.0

lr=0.0005
lradj=type1
train_epochs=10
patience=3
batch_size=32

rerun=0

for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${input_trans}_${chan_indep}_${extra_rev_in}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 358 \
            --dec_in 358 \
            --c_out 358 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done








# hyper-parameters
dst=PEMS08
pl_list=(12 24 36 48)
extra_rev_in_list=(1 0)
input_trans_list=(evd)

lambda=1.0

lr=0.0005
lradj=type1
train_epochs=10
patience=3
batch_size=32

rerun=0

for input_trans_item in ${input_trans_list[@]}; do
case $input_trans_item in
    evd) chan_indep_list=(0) input_trans=$input_trans_item;;
    same) chan_indep_list=(0) input_trans=$auxi_type;;
    *) chan_indep_list=(0) input_trans=$input_trans_item;;
esac
for chan_indep in ${chan_indep_list[@]}; do
for extra_rev_in in ${extra_rev_in_list[@]}; do
for pl in ${pl_list[@]}; do
    if ! [[ " ${datasets[@]} " =~ " ${dst} " ]]; then
        continue
    fi

    rl=$lambda
    ax=$(echo "1 - $lambda" | bc)
    decimal_places=$(echo "$lambda" | awk -F. '{print length($2)}')
    ax=$(printf "%.${decimal_places}f" $ax)

    JOB_NAME=${model_name}_${dst}_${pl}_${input_trans}_${chan_indep}_${extra_rev_in}_${rl}_${ax}_${lr}_${lradj}_${train_epochs}_${patience}_${batch_size}
    OUTPUT_DIR="${OUT_ROOT}/results/${EXP_NAME}/${JOB_NAME}"
    PROJ_DIR="${OUT_ROOT}/projections/PCA/${dst}"
    input_trans_path="${OUT_ROOT}/projections/EVD/${dst}/ci${chan_indep}"
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
            --task_name long_term_forecast_trans \
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
            --enc_in 170 \
            --dec_in 170 \
            --c_out 170 \
            --e_layers 2 \
            --d_layers 1 \
            --factor 3 \
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
            --fix_seed ${seed} \
            --checkpoints $CHECKPOINTS \
            --results $RESULTS \
            --test_results $TEST_RESULTS \
            --log_path $LOG_PATH \
            --rerun $rerun \
            --load_from_disk ${PROJ_DIR} \
            --input_trans ${input_trans} \
            --input_trans_path ${input_trans_path} \
            --chan_indep ${chan_indep} \
            --extra_rev_in ${extra_rev_in} \
            --speedup_sklearn 2

        sleep 5
    } 2>&1 | tee -a "${OUTPUT_DIR}/stdout.log" &
done
done
done
done




wait