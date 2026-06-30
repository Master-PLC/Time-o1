#!/bin/bash

BASE_PATH="/mnt/tidalfs-bdsz01/usr/panlicheng/workspace/ProjDF-Meta/results_ML3/finetune"
BASE_PATH="/mnt/tidalfs-bdsz01/usr/panlicheng/workspace/ProjDF-Meta/results_ML3/diff_loss"
BASE_PATH="/mnt/tidalfs-bdsz01/usr/panlicheng/workspace/ProjDF-Meta/results_ML3/diff_meta"
BASE_PATH="/mnt/tidalfs-bdsz01/usr/panlicheng/workspace/ProjDF-Meta/results_ML3/diff_seed"
BASE_PATH="/mnt/tidalfs-bdsz01/usr/panlicheng/workspace/ProjDF-Meta/results_ML3/vary_input"

# 最大并发数
MAX_JOBS=32
TEMP_DIR=$(mktemp -d)

# 当前运行的作业数
running_jobs=0

# 单个文件夹处理函数
process_folder_bg() {
    local folder="$1"
    local job_id="$2"
    local folder_name=$(basename "$folder")
    local result_file="$TEMP_DIR/job_$job_id"
    
    {
        # 查找匹配文件
        mapfile -t files < <(find "$folder" -type f \( -name "*.txt" -o -name "*.log" -o -name "*.pdf" -o -name "events*" -o -name "*.pth" \))
        
        if [ ${#files[@]} -eq 0 ]; then
            echo "0|0|$folder_name|No files" > "$result_file"
            return 0
        fi
        
        # 创建压缩包
        archive_path="$folder/archive.tar.xz"
        temp_list=$(mktemp)
        
        for file in "${files[@]}"; do
            echo "${file#$folder/}" >> "$temp_list"
        done
        
        # 打包
        if tar -cJf "$archive_path" -C "$folder" -T "$temp_list" > /dev/null 2>&1; then
            # 删除原文件
            for file in "${files[@]}"; do
                rm -f "$file"
            done
            echo "1|${#files[@]}|$folder_name|✓ $folder_name: ${#files[@]} files archived" > "$result_file"
        else
            echo "0|0|$folder_name|✗ $folder_name: Error occurred" > "$result_file"
        fi
        
        rm -f "$temp_list"
    } &
}

# 等待作业完成
wait_for_jobs() {
    local max_wait=${1:-0}
    
    while [ $running_jobs -gt $max_wait ]; do
        wait -n 2>/dev/null
        ((running_jobs--))
    done
}

# 处理所有文件夹
job_counter=0
for folder in "$BASE_PATH"/*; do
    [ ! -d "$folder" ] && continue
    
    # 如果达到最大并发数，等待一个作业完成
    if [ $running_jobs -ge $MAX_JOBS ]; then
        wait_for_jobs $((MAX_JOBS-1))
    fi
    
    # 仅当有文件需要处理时才显示开始信息
    if [ $(find "$folder" -type f \( -name "*.txt" -o -name "*.log" -o -name "*.pdf" -o -name "events*" -o -name "*.pth" \) | wc -l) -gt 0 ]; then
        echo "[$(date +'%H:%M:%S')] Starting: $(basename "$folder")"
    fi
    
    process_folder_bg "$folder" "$job_counter"
    ((running_jobs++))
    ((job_counter++))
done

# 等待所有作业完成
wait_for_jobs 0

# 汇总结果
total_processed=0
total_files=0

echo -e "\nProcessing results:"
for i in $(seq 0 $((job_counter-1))); do
    result_file="$TEMP_DIR/job_$i"
    [ -f "$result_file" ] || continue
    
    IFS='|' read -r processed files folder_name message < "$result_file"
    
    if [ "$processed" -eq 1 ]; then
        ((total_processed++))
        ((total_files+=files))
        echo "$message"
    elif [[ "$message" == *"Error"* ]]; then
        echo "$message"
    fi
done

echo ""
echo "Summary: $total_processed folders processed, $total_files files archived"

# 清理
rm -rf "$TEMP_DIR"
