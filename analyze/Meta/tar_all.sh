#!/bin/bash

BASE_PATH="/mnt/tidalfs-bdsz01/usr/panlicheng/workspace/ProjDF-Meta/results_ML3/finetune"

total_processed=0
total_files=0

for folder in "$BASE_PATH"/*; do
    [ ! -d "$folder" ] && continue
    
    folder_name=$(basename "$folder")
    
    # 查找匹配文件
    mapfile -t files < <(find "$folder" -type f \( -name "*.txt" -o -name "*.log" -o -name "*.pdf" -o -name "events*" -o -name "*.pth" \))
    
    [ ${#files[@]} -eq 0 ] && continue
    
    # 创建压缩包
    archive_path="$folder/archive.tar.xz"
    
    # 创建临时文件列表（相对路径）
    temp_list=$(mktemp)
    for file in "${files[@]}"; do
        echo "${file#$folder/}" >> "$temp_list"
    done
    
    # 打包
    if tar -cJf "$archive_path" -C "$folder" -T "$temp_list" 2>/dev/null; then
        # 删除原文件
        for file in "${files[@]}"; do
            rm "$file"
        done
        
        echo "✓ $folder_name: ${#files[@]} files archived"
        ((total_processed++))
        ((total_files+=${#files[@]}))
    else
        echo "✗ $folder_name: Error occurred"
    fi
    
    rm "$temp_list"
done

echo ""
echo "Summary: $total_processed folders processed, $total_files files archived"
