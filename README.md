# Parallel Programming HW4 - CUDA Bitcoin Miner

## 專案概述

本專案實作了一個使用 CUDA 平行化的比特幣挖礦程式，透過 GPU 加速尋找符合目標難度的 nonce 值。

## 實作方法

### 平行化策略

- 將原本序列版本的 for-loop 任務分配給多個 GPU 執行緒平行處理
- 每個執行緒測試不同的 nonce 值（`global_tid`, `global_tid + thread_count`, ...）
- 使用全域記憶體中的 flag 標記是否已找到目標 nonce
- 當任一執行緒找到答案時，所有執行緒能即時偵測並提前終止

### 記憶體優化

1. **Constant Memory**
   - 使用 `cudaMemcpyToSymbol` 將 block template 和 target difficulty 複製至 constant memory
   - 提供高效的唯讀存取

2. **SHA-256 最佳化**
   - 針對固定長度（80 bytes 和 32 bytes）特化 SHA-256 函數，移除不必要的長度檢查
   - 將原本 64 個 words 的陣列改為 16 個 words 的循環緩衝區，使其能放入暫存器
   - 展開迴圈並使用純量變數，消除陣列索引計算

3. **Midstate 預計算**
   - 在 CPU 端預先計算區塊頭不變部分的 hash（midstate）
   - 每個執行緒從 midstate 繼續計算，省略一次 `sha256_transform`
   - 大幅提升整體效能

## 編譯與執行

### 編譯

```bash
cd b10705009
make
```

### 執行

```bash
./hw4 <input_file> <output_file>
```

範例：
```bash
./hw4 ../testcases/case00.in output.out
```

### 清理

```bash
make clean
```

## 檔案結構

```
b10705009/
├── hw4.cu           # 主程式（包含 CUDA kernel 與 host code）
├── sha256.cu        # SHA-256 實作（CPU 與 GPU 版本）
├── sha256.h         # SHA-256 標頭檔
├── Makefile         # 編譯設定
└── report.pdf       # 實驗報告
```

