#!/usr/bin/env bash
set -euo pipefail

# 要跑的測資編號
TESTS=("00" "01" "02" "03")

# 初始化總時間
GRAND_TOTAL_TIME=0

# summary header
echo "case correctness total_time_ms occupancy_pct"

for t in "${TESTS[@]}"; do
    in_file="testcases/case${t}.in"
    out_file="output/case${t}.out"
    log_file="ncu_case${t}.log"

    # 1. 使用 Nsight Compute 收集 occupancy + 時間 (輸出存到 log)
    # 如果在 cluster 需要 srun/sbatch，在這行外面包即可。
    # ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
        ./sample/hw4 "$in_file" "$out_file" >"$log_file" 2>&1

    # 2. correctness：看 validation 有沒有出現 "correct"
    val_output="$(./validation "$in_file" "$out_file" 2>&1 || true)"
    if echo "$val_output" | grep -qi "correct"; then
        correctness="correct"
    else
        correctness="wrong"
    fi

    # 3. total time：加總所有 [Time] ===== Total time: X ms =====
    # 支援多個 task 的情況
    times="$(grep '\[Time\] ===== Total time:' "$log_file" | awk '{print $5}' || true)"
    if [[ -z "${times}" ]]; then
        total_time_ms="0"
    else
        total_time_ms="$(awk '{sum += $1} END {print sum}' <<< "$times")"
    fi

    # 4. occupancy：取最後一行 metric 的數字
    occ_line="$(grep 'sm__warps_active.avg.pct_of_peak_sustained_active' "$log_file" | tail -n 1 || true)"
    if [[ -n "${occ_line}" ]]; then
        occupancy_pct="$(awk '{print $NF}' <<< "$occ_line")"
    else
        occupancy_pct="NA"
    fi

    # 5. 印出 summary
    echo "case${t} ${correctness} ${total_time_ms} ${occupancy_pct}"
    
    # 累加總時間
    GRAND_TOTAL_TIME=$(awk "BEGIN {print $GRAND_TOTAL_TIME + $total_time_ms}")
done

# 印出所有測資的總時間
echo ""
echo "=========================================="
echo "Total time (all cases): ${GRAND_TOTAL_TIME} ms"
echo "=========================================="
