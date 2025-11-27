#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned int  WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx {
    WORD h[8];
    BYTE b[32];
} SHA256;

// CPU
void sha256_transform_cpu(SHA256 *ctx, const BYTE *msg);
void sha256_cpu(SHA256 *ctx, const BYTE *msg, size_t len);

// GPU
__device__ void sha256_transform_gpu(SHA256 *ctx, const BYTE *msg);
__device__ void sha256_gpu(SHA256 *ctx, const BYTE *msg, size_t len);

// 通用 double sha256（任意長度訊息）
__device__ void double_sha256_gpu(SHA256 *out, const BYTE *bytes, size_t len);

// 專門給 80-byte Bitcoin block header 的 double sha256
__device__ void double_sha256_bitcoin_specialized(SHA256 *out, const BYTE *block80);

// 計算第一輪 SHA 的 midstate：只吃 64 bytes，不做 padding
void sha256_midstate_cpu(const BYTE *block64, WORD midstate[8]);

// 從 midstate + block tail (含 nonce) 做 double-SHA256（給 GPU kernel 用）
__device__ void double_sha256_from_midstate(SHA256 *out,
                                            const WORD midstate[8],
                                            const BYTE *block80);

#ifdef __cplusplus
}
#endif

#endif // __SHA256_HEADER__
