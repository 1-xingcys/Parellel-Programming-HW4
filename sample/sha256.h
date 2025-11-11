#ifndef __SHA256_HEADER__
#define __SHA256_HEADER__

#include <stddef.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

//--------------- DATA TYPES --------------
typedef unsigned int WORD;
typedef unsigned char BYTE;

typedef union _sha256_ctx{
	WORD h[8];
	BYTE b[32];
}SHA256;

//----------- FUNCTION DECLARATION --------
void sha256_transform_cpu(SHA256 *ctx, const BYTE *msg);
void sha256_cpu(SHA256 *ctx, const BYTE *msg, size_t len);

__device__ void sha256_transform_gpu(SHA256 *ctx, const BYTE *msg);
__device__ void sha256_gpu(SHA256 *ctx, const BYTE *msg, size_t len);
__device__ void double_sha256_bitcoin_specialized(SHA256 *final_hash_ctx, const BYTE *block_80_bytes);
__device__ void double_sha256_gpu(SHA256 *sha256_ctx, const BYTE *bytes, size_t len);


#ifdef __cplusplus
}
#endif  //__cplusplus

#endif  //__SHA256_HEADER__
