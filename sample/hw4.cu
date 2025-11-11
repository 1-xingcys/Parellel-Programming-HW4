//***********************************************************************************
// 2018.04.01 created by Zexlus1126
//
//    Example 002
// This is a simple demonstration on calculating merkle root from merkle branch 
// and solving a block (#286819) which the information is downloaded from Block Explorer 
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>

#include <chrono>

#include "sha256.h"

#include <cuda_runtime.h>

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;


////////////////////////   Utils   ///////////////////////

//convert one hex-codec char to binary
unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
    return 0;
}


// convert hex string to binary
//
// in: input string
// string_len: the length of the input string
//      '\0' is not included in string_len!!!
// out: output bytes array
void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

// print out binary array (from highest value) in the hex format
void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}


// print out binar array (from lowest value) in the hex format
void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

int __device__ __host__ little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    // compared from lowest bit
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{

    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256_cpu(&tmp, (BYTE*)bytes, len);
    sha256_cpu(sha256_ctx, (BYTE*)&tmp, 32);
}


////////////////////   Merkle Root   /////////////////////


// calculate merkle root from several merkle branches
// root: output hash will store here (little-endian)
// branch: merkle branch  (big-endian)
// count: total number of merkle branch
void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count; // merkle branch
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    // copy each branch to the list
    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        //convert hex string to bytes array and store them into the list
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;


    // calculate merkle root
    while(total_count > 1)
    {
        
        // hash each pair
        int i, j;

        if(total_count % 2 == 1)  //odd, 
        {
            memcpy(list[total_count], list[total_count-1], 32);
            ++total_count;
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            // this part is slightly tricky,
            //   because of the implementation of the double_sha256,
            //   we can avoid the memory begin overwritten during our sha256d calculation
            // double_sha:
            //     tmp = hash(list[0]+list[1])
            //     list[0] = hash(tmp)
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

// ******************************************************************
// ** CUDA 挖礦核心 (Kernel)
// ******************************************************************
//
// g_block_template: 80-byte 的區塊頭模板 (nonce 會被覆蓋)
// g_target_hex:     32-byte 的目標值
//
__constant__ HashBlock g_block_template;
__constant__ unsigned char g_target_hex[32];


//
// GPU 上的挖礦核心
//
// d_solution_nonce: 一個指向 GPU 全域記憶體的指標。
//                  如果一個執行緒找到了答案，它會把 nonce 寫入這裡。
//                  它被初始化為 0xFFFFFFFF (代表 "未找到")。
//
__global__ void solve_kernel(volatile unsigned int *d_solution_nonce)
{
    // --- 計算這個執行緒要處理的 nonce ---
    // 使用 Grid-Stride Loop
    unsigned long long int start_nonce = (unsigned long long int)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long int stride = (unsigned long long int)gridDim.x * blockDim.x;

    // --- 準備本地資料 ---
    // 從快速的 __constant__ 記憶體中複製一份區塊模板
    HashBlock local_block = g_block_template;
    SHA256 local_hash_ctx; // 用於儲存 hash 結果

    for (unsigned long long n = start_nonce; n <= 0xFFFFFFFF; n += stride)
    {
        // --- 檢查是否有人找到了 ---
        if (*d_solution_nonce != 0xFFFFFFFF)
        {
            return;
        }

        // --- 執行工作 ---
        // 填入這個執行緒要測試的 nonce
        local_block.nonce = (unsigned int)n;
        
        // 執行 Device 上的 double_sha256_gpu
        // 輸入是 80-byte 的 local_block
        // 輸出是 32-byte 的 hash，儲存在 local_hash_ctx.b
        // double_sha256_gpu(&local_hash_ctx, (unsigned char*)&local_block, sizeof(local_block));
        double_sha256_bitcoin_specialized(&local_hash_ctx, (unsigned char*)&local_block);

        // --- 檢查答案 ---
        // 比較 hash (local_hash_ctx.b) 是否小於目標 (g_target_hex)
        if (little_endian_bit_comparison(local_hash_ctx.b, g_target_hex, 32) < 0)
        {
            // --- 找到了！回報答案 ---
            // 使用 atomicCAS 確保只有「第一個」找到答案的執行緒
            // 可以成功寫入。
            // 參數: (目標地址, 舊值, 新值)
            // 只有當 *d_solution_nonce 仍為 0xFFFFFFFF 時，
            // 才將它設為 n。
            atomicCAS((unsigned int*)d_solution_nonce, 0xFFFFFFFF, (unsigned int)n);
            return;
        }
    }
}


// ******************************************************************
// ** 修改後的 `solve` 和 `main` 函數 (Host Code)
// ******************************************************************

//
// solve (CPU Host Code)
// 負責：
// 1. 執行所有 CPU 端的設定 (同前)
// 2. 設定 CUDA
// 3. 啟動 `solve_kernel`
// 4. 取得結果
//
void solve(FILE *fin, FILE *fout)
{
    // 計時器
    // auto total_start = std::chrono::high_resolution_clock::now();
    
    // **** 讀取資料 ****
    // auto stage_start = std::chrono::high_resolution_clock::now();
    
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }
    
    // auto stage_end = std::chrono::high_resolution_clock::now();
    // auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
    // printf("[Time] Read input data: %.3f ms\n", read_time / 1000.0);

    // **** 計算 Merkle Root ****
    // stage_start = std::chrono::high_resolution_clock::now();
    
    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);
    
    // stage_end = std::chrono::high_resolution_clock::now();
    // auto merkle_time = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
    // printf("[Time] Calculate Merkle Root: %.3f ms\n", merkle_time / 1000.0);

    // printf("merkle root(big):    ");
    // print_hex_inverse(merkle_root, 32);
    // printf("\n");

    // **** 準備 Block Header 模板 ****
    // stage_start = std::chrono::high_resolution_clock::now();
    HashBlock block_template;
    convert_string_to_little_endian_bytes((unsigned char *)&block_template.version, version, 8);
    convert_string_to_little_endian_bytes(block_template.prevhash,                  prevhash,    64);
    memcpy(block_template.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block_template.nbits,   nbits,     8);
    convert_string_to_little_endian_bytes((unsigned char *)&block_template.ntime,   ntime,     8);
    block_template.nonce = 0; // Kernel 會覆寫它
    
    
    // **** 計算 Target Value ****
    unsigned int exp = block_template.nbits >> 24;
    unsigned int mant = block_template.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    // little-endian
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));
    
    // stage_end = std::chrono::high_resolution_clock::now();
    // auto prepare_time = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
    // printf("[Time] Prepare block header and target: %.3f ms\n", prepare_time / 1000.0);
    
    // printf("Target value (big): ");
    // print_hex_inverse(target_hex, 32);
    // printf("\n");

    // ********** CUDA 執行 **********
    
    // 建立 CUDA Event 用於精確計時
    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);
    
    // **** CUDA 記憶體配置 ****
    // stage_start = std::chrono::high_resolution_clock::now();
    
    // 在 GPU 上配置記憶體，用於接收答案
    unsigned int *d_solution_nonce;
    unsigned int h_solution_nonce = 0xFFFFFFFF; // "未找到" 的初始值
    
    cudaError_t err = cudaMalloc((void**)&d_solution_nonce, sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // 2. 將 GPU 上的答案記憶體初始化為 "未找到"
    cudaMemcpy(d_solution_nonce, &h_solution_nonce, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // 3. 將 Block 模板和 Target 複製到 __constant__ 記憶體
    cudaMemcpyToSymbol(g_block_template, &block_template, sizeof(HashBlock));
    cudaMemcpyToSymbol(g_target_hex, target_hex, 32);
    
    // stage_end = std::chrono::high_resolution_clock::now();
    // auto mem_alloc_time = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
    // printf("[Time] CUDA memory allocation and copy: %.3f ms\n", mem_alloc_time / 1000.0);

    
    // ********** 啟動 Kernel **************
    
    // 配置執行緒網格 (Grid) 和區塊 (Block)
    int threadsPerBlock = 192;
    int blocksPerGrid = 80 * 32;

    // printf("Starting CUDA kernel (Threads: %d, Blocks: %d) to find nonce...\n", threadsPerBlock, blocksPerGrid);
    
    // 開始計時 Kernel 執行
    cudaEventRecord(cuda_start);
    
    solve_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_solution_nonce);

    // 等待 Kernel 執行完畢
    cudaDeviceSynchronize();
    
    // 結束計時 Kernel 執行
    cudaEventRecord(cuda_end);
    cudaEventSynchronize(cuda_end);
    
    // float kernel_time_ms = 0;
    // cudaEventElapsedTime(&kernel_time_ms, cuda_start, cuda_end);
    // printf("[Time] Kernel execution: %.3f ms\n", kernel_time_ms);
    
    // 檢查 Kernel 啟動是否有錯誤
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_solution_nonce);
        cudaEventDestroy(cuda_start);
        cudaEventDestroy(cuda_end);
        return;
    }

    // ********** 取得結果 **************
    // 將答案 (或 0xFFFFFFFF) 從 GPU 複製回 CPU
    // stage_start = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(&h_solution_nonce, d_solution_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // stage_end = std::chrono::high_resolution_clock::now();
    // auto copy_back_time = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
    // printf("[Time] Copy result back to CPU: %.3f ms\n", copy_back_time / 1000.0);

    // ********** 顯示與清理 **************

    if (h_solution_nonce != 0xFFFFFFFF)
    {
        // printf("Found Solution!!\n");
        // printf("Nonce: %u (0x%x)\n", h_solution_nonce, h_solution_nonce);
        
        // 驗證 (在 CPU 上重算一次，確保 GPU 沒算錯)
        // stage_start = std::chrono::high_resolution_clock::now();
        // block_template.nonce = h_solution_nonce;
        // SHA256 sha256_ctx;
        // double_sha256(&sha256_ctx, (unsigned char*)&block_template, sizeof(block_template));
        
        // stage_end = std::chrono::high_resolution_clock::now();
        // auto verify_time = std::chrono::duration_cast<std::chrono::microseconds>(stage_end - stage_start).count();
        // printf("[Time] CPU verification: %.3f ms\n", verify_time / 1000.0);
        
        // printf("Verified Hash (big): ");
        // print_hex_inverse(sha256_ctx.b, 32);
        // printf("\n");
        
        // 寫入檔案
        for(int i=0;i<4;++i)
        {
            fprintf(fout, "%02x", ((unsigned char*)&h_solution_nonce)[i]);
        }
        fprintf(fout, "\n");
    }
    else
    {
        printf("No solution found in the 32-bit nonce space.\n");
    }
    
    // 計算總時間
    // auto total_end = std::chrono::high_resolution_clock::now();
    // auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    // printf("[Time] ===== Total time: %.3f ms =====\n", total_time / 1000.0);
    // printf("\n");

    // 清理 GPU 記憶體和 Event
    cudaFree(d_solution_nonce);
    cudaEventDestroy(cuda_start);
    cudaEventDestroy(cuda_end);
    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

// main (CPU Host Code)
int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: cuda_miner <in> <out>\n");
        return 1;
    }
    FILE *fin = fopen(argv[1], "r");
    if (!fin) {
        fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
        return 1;
    }
    FILE *fout = fopen(argv[2], "w");
    if (!fout) {
        fprintf(stderr, "Error: Cannot open output file %s\n", argv[2]);
        fclose(fin);
        return 1;
    }

    int totalblock;

    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i)
    {
        // printf("--- Solving Block %d ---\n", i+1);
        solve(fin, fout);
    }
    
    fclose(fin);
    fclose(fout);

    return 0;
}