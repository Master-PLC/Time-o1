out_root=/mnt/tidalfs-bdsz01/dataset/llm_ckpt/plc_data/Time-o1/projections/PCA

for dst in ETTh1 ETTh2 ETTm1 ETTm2 ECL Weather Traffic PEMS03 PEMS08; do
    mkdir -p ${out_root}/${dst}/sp2/input/T
    mkdir -p ${out_root}/${dst}/sp2/input_mark/T
    mkdir -p ${out_root}/${dst}/sp2/output/T

    mv ${out_root}/${dst}/sp2/input/96 ${out_root}/${dst}/sp2/input/T/
    mv ${out_root}/${dst}/sp2/input_mark/96 ${out_root}/${dst}/sp2/input_mark/T/
    case ${dst} in
        PEMS03 | PEMS08) pl_list=(12 24 36 48);;
        *) pl_list=(96 192 336 720);;
    esac
    for pl in ${pl_list[@]}; do
        mv ${out_root}/${dst}/sp2/output/${pl} ${out_root}/${dst}/sp2/output/T/
    done
done