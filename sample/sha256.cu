
// This sha256 implementation is based on sha256 wiki page
// please refer to:
//     https://en.wikipedia.org/wiki/SHA-2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include <cuda_runtime.h>

// circular shift - wiki:
//     https://en.wikipedia.org/wiki/Circular_shift
#define _rotl(v, s) ((v)<<(s) | (v)>>(32-(s)))
#define _rotr(v, s) ((v)>>(s) | (v)<<(32-(s)))

#define _swap_cpu(x, y) (((x)^=(y)), ((y)^=(x)), ((x)^=(y)))

#ifdef __cplusplus
extern "C"{
#endif  //__cplusplus

// ===================== GPU helpers =====================

__device__ __forceinline__ void _swap_gpu(BYTE &x, BYTE &y) {
    BYTE t = x;
    x = y;
    y = t;
}

__device__ __forceinline__ WORD rotr_gpu(WORD v, int s) {
#if __CUDA_ARCH__ >= 350
    // funnel shift: rotate right s bits
    return __funnelshift_r(v, v, s);
#else
    return (v >> s) | (v << (32 - s));
#endif
}

// ===================== Constants =======================

static const WORD k_cpu[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// 放在 constant memory（GPU 廣播快取）
__constant__ WORD k_gpu[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// ===================== GPU sha256 core =======================

// 使用 16-word 環形 buffer 的 transform：省暫存器、穩健快速
__device__ __forceinline__ void sha256_transform_gpu(SHA256 *ctx, const BYTE *msg)
{
    WORD w[16];

    // 前 16 個 word
#pragma unroll
    for (int i = 0, j = 0; i < 16; ++i, j += 4) {
        w[i] = ( (WORD)msg[j]   << 24 ) |
               ( (WORD)msg[j+1] << 16 ) |
               ( (WORD)msg[j+2] <<  8 ) |
               ( (WORD)msg[j+3]       );
    }

    WORD a = ctx->h[0];
    WORD b = ctx->h[1];
    WORD c = ctx->h[2];
    WORD d = ctx->h[3];
    WORD e = ctx->h[4];
    WORD f = ctx->h[5];
    WORD g = ctx->h[6];
    WORD h = ctx->h[7];

#pragma unroll
    for (int i = 0; i < 64; ++i) {
        WORD Wt;
        if (i < 16) {
            Wt = w[i];
        } else {
            WORD w15 = w[(i - 15) & 15];
            WORD w2  = w[(i - 2)  & 15];
            WORD s0 = rotr_gpu(w15, 7) ^ rotr_gpu(w15, 18) ^ (w15 >> 3);
            WORD s1 = rotr_gpu(w2, 17) ^ rotr_gpu(w2, 19)  ^ (w2 >> 10);
            Wt = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
            w[i & 15] = Wt;
        }

        WORD S1  = rotr_gpu(e, 6) ^ rotr_gpu(e, 11) ^ rotr_gpu(e, 25);
        WORD ch  = (e & f) ^ ((~e) & g);
        WORD temp1 = h + S1 + ch + k_gpu[i] + Wt;

        WORD S0  = rotr_gpu(a, 2) ^ rotr_gpu(a, 13) ^ rotr_gpu(a, 22);
        WORD maj = (a & b) ^ (a & c) ^ (b & c);
        WORD temp2 = S0 + maj;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

// 通用 GPU 版 sha256（含 padding）
__device__ void sha256_gpu(SHA256 *ctx, const BYTE *msg, size_t len)
{
    // init
    ctx->h[0] = 0x6a09e667;
    ctx->h[1] = 0xbb67ae85;
    ctx->h[2] = 0x3c6ef372;
    ctx->h[3] = 0xa54ff53a;
    ctx->h[4] = 0x510e527f;
    ctx->h[5] = 0x9b05688c;
    ctx->h[6] = 0x1f83d9ab;
    ctx->h[7] = 0x5be0cd19;

    size_t i = 0;

    // 處理完整 64-byte block
    while (i + 64 <= len) {
        sha256_transform_gpu(ctx, msg + i);
        i += 64;
    }

    // 準備最後一個（或兩個） block
    BYTE block[64];

    size_t rem = len - i;
#pragma unroll
    for (size_t j = 0; j < rem; ++j) {
        block[j] = msg[i + j];
    }

    block[rem++] = 0x80;  // append '1' bit

    if (rem > 56) {
        // 填 0 到 64
#pragma unroll
        for (size_t j = rem; j < 64; ++j) block[j] = 0;
        sha256_transform_gpu(ctx, block);
        rem = 0;
    }

    // 填 0 到 56
#pragma unroll
    for (size_t j = rem; j < 56; ++j) block[j] = 0;

    unsigned long long bitlen = (unsigned long long)len * 8ULL;
    block[63] = (BYTE)(bitlen      );
    block[62] = (BYTE)(bitlen >> 8 );
    block[61] = (BYTE)(bitlen >>16 );
    block[60] = (BYTE)(bitlen >>24 );
    block[59] = (BYTE)(bitlen >>32 );
    block[58] = (BYTE)(bitlen >>40 );
    block[57] = (BYTE)(bitlen >>48 );
    block[56] = (BYTE)(bitlen >>56 );

    sha256_transform_gpu(ctx, block);

    // output big-endian bytes (跟 CPU 版一致)
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
        _swap_gpu(ctx->b[j],   ctx->b[j+3]);
        _swap_gpu(ctx->b[j+1], ctx->b[j+2]);
    }
}

// 專門給 80-byte 區塊頭用的 Double SHA-256 特化版
// 輸入：block_80_bytes = 區塊頭序列化後的 80 bytes
// 輸出：final_hash_ctx->b 內為「大端」32-byte 雜湊值（和一般實作一致）
__device__
void double_sha256_bitcoin_specialized(SHA256 *final_hash_ctx, const BYTE *block_80_bytes)
{
    // ---------- Round 1: SHA256( block_header[80] ) ----------

    SHA256 r1_ctx;
    BYTE   r1_chunk2[64];

    // 初始化 Round 1 狀態
    r1_ctx.h[0] = 0x6a09e667; r1_ctx.h[1] = 0xbb67ae85;
    r1_ctx.h[2] = 0x3c6ef372; r1_ctx.h[3] = 0xa54ff53a;
    r1_ctx.h[4] = 0x510e527f; r1_ctx.h[5] = 0x9b05688c;
    r1_ctx.h[6] = 0x1f83d9ab; r1_ctx.h[7] = 0x5be0cd19;

    // Block 0: 前 64 bytes
    sha256_transform_gpu(&r1_ctx, block_80_bytes);

    // Block 1: 剩下 16 bytes + padding + 長度(640 bits)
    // 先放 16 bytes
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        r1_chunk2[i] = block_80_bytes[64 + i];
    }

    // 接著是 0x80
    r1_chunk2[16] = 0x80;

    // 填零到第 55 位
#pragma unroll
    for (int i = 17; i < 56; ++i) {
        r1_chunk2[i] = 0x00;
    }

    // 最後 8 bytes = 80 * 8 = 640 bits = 0x0000000000000280 (big-endian)
    r1_chunk2[56] = 0x00;
    r1_chunk2[57] = 0x00;
    r1_chunk2[58] = 0x00;
    r1_chunk2[59] = 0x00;
    r1_chunk2[60] = 0x00;
    r1_chunk2[61] = 0x00;
    r1_chunk2[62] = 0x02;
    r1_chunk2[63] = 0x80;

    // 處理 Block 1
    sha256_transform_gpu(&r1_ctx, r1_chunk2);

    // Round 1 結束：把 state 轉成 Big-Endian 32-byte digest（放在 r1_ctx.b）
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        _swap_gpu(r1_ctx.b[i],     r1_ctx.b[i + 3]);
        _swap_gpu(r1_ctx.b[i + 1], r1_ctx.b[i + 2]);
    }

    // ---------- Round 2: SHA256( Round1_digest[32] ) ----------

    BYTE r2_block[64];

    // 初始化 Round 2 狀態
    final_hash_ctx->h[0] = 0x6a09e667; final_hash_ctx->h[1] = 0xbb67ae85;
    final_hash_ctx->h[2] = 0x3c6ef372; final_hash_ctx->h[3] = 0xa54ff53a;
    final_hash_ctx->h[4] = 0x510e527f; final_hash_ctx->h[5] = 0x9b05688c;
    final_hash_ctx->h[6] = 0x1f83d9ab; final_hash_ctx->h[7] = 0x5be0cd19;

    // 前 32 bytes = Round 1 digest
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        r2_block[i] = r1_ctx.b[i];
    }

    // padding: 0x80
    r2_block[32] = 0x80;

    // 填零到第 55 位
#pragma unroll
    for (int i = 33; i < 56; ++i) {
        r2_block[i] = 0x00;
    }

    // 長度 = 32 * 8 = 256 bits = 0x0000000000000100 (big-endian)
    r2_block[56] = 0x00;
    r2_block[57] = 0x00;
    r2_block[58] = 0x00;
    r2_block[59] = 0x00;
    r2_block[60] = 0x00;
    r2_block[61] = 0x00;
    r2_block[62] = 0x01;
    r2_block[63] = 0x00;

    // 單一 block
    sha256_transform_gpu(final_hash_ctx, r2_block);

    // 最終輸出：轉成 Big-Endian（和 CPU sha256_cpu 最後一步對齊）
#pragma unroll
    for (int i = 0; i < 32; i += 4) {
        _swap_gpu(final_hash_ctx->b[i],     final_hash_ctx->b[i + 3]);
        _swap_gpu(final_hash_ctx->b[i + 1], final_hash_ctx->b[i + 2]);
    }
}


// 正確版 double_sha256_gpu：第二輪只吃 32 bytes digest
__device__ void double_sha256_gpu(SHA256 *sha256_ctx, const BYTE *bytes, size_t len)
{
    SHA256 tmp;
    sha256_gpu(&tmp, bytes, len);
    sha256_gpu(sha256_ctx, tmp.b, 32);  // 只對 32-byte hash 做第二輪
}

void sha256_transform_cpu(SHA256 *ctx, const BYTE *msg)
{
	WORD a, b, c, d, e, f, g, h;
	WORD i, j;
	
	WORD w[64];
	for(i=0, j=0;i<16;++i, j+=4)
	{
		w[i] = (msg[j]<<24) | (msg[j+1]<<16) | (msg[j+2]<<8) | (msg[j+3]);
	}
	
	for(i=16;i<64;++i)
	{
		WORD s0 = (_rotr(w[i-15], 7)) ^ (_rotr(w[i-15], 18)) ^ (w[i-15]>>3);
		WORD s1 = (_rotr(w[i-2], 17)) ^ (_rotr(w[i-2], 19))  ^ (w[i-2]>>10);
		w[i] = w[i-16] + s0 + w[i-7] + s1;
	}
	
	a = ctx->h[0];
	b = ctx->h[1];
	c = ctx->h[2];
	d = ctx->h[3];
	e = ctx->h[4];
	f = ctx->h[5];
	g = ctx->h[6];
	h = ctx->h[7];
	
	for(i=0;i<64;++i)
	{
		WORD S0 = (_rotr(a, 2)) ^ (_rotr(a, 13)) ^ (_rotr(a, 22));
		WORD S1 = (_rotr(e, 6)) ^ (_rotr(e, 11)) ^ (_rotr(e, 25));
		WORD ch = (e & f) ^ ((~e) & g);
		WORD maj = (a & b) ^ (a & c) ^ (b & c);
		WORD temp1 = h + S1 + ch + k_cpu[i] + w[i];
		WORD temp2 = S0 + maj;
		
		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}
	
	ctx->h[0] += a;
	ctx->h[1] += b;
	ctx->h[2] += c;
	ctx->h[3] += d;
	ctx->h[4] += e;
	ctx->h[5] += f;
	ctx->h[6] += g;
	ctx->h[7] += h;
}

void sha256_cpu(SHA256 *ctx, const BYTE *msg, size_t len)
{
	ctx->h[0] = 0x6a09e667;
	ctx->h[1] = 0xbb67ae85;
	ctx->h[2] = 0x3c6ef372;
	ctx->h[3] = 0xa54ff53a;
	ctx->h[4] = 0x510e527f;
	ctx->h[5] = 0x9b05688c;
	ctx->h[6] = 0x1f83d9ab;
	ctx->h[7] = 0x5be0cd19;
	
	WORD i, j;
	size_t remain = len % 64;
	size_t total_len = len - remain;
	
	for(i=0;i<total_len;i+=64)
	{
		sha256_transform_cpu(ctx, &msg[i]);
	}
	
	BYTE m[64] = {};
	for(i=total_len, j=0;i<len;++i, ++j)
	{
		m[j] = msg[i];
	}
	
	m[j++] = 0x80;
	
	if(j > 56)
	{
		sha256_transform_cpu(ctx, m);
		memset(m, 0, sizeof(m));
	}
	
	unsigned long long L = (unsigned long long)len * 8ULL;
	m[63] = L;
	m[62] = L >> 8;
	m[61] = L >> 16;
	m[60] = L >> 24;
	m[59] = L >> 32;
	m[58] = L >> 40;
	m[57] = L >> 48;
	m[56] = L >> 56;
	sha256_transform_cpu(ctx, m);
	
	for(i=0;i<32;i+=4)
	{
        _swap_cpu(ctx->b[i],   ctx->b[i+3]);
        _swap_cpu(ctx->b[i+1], ctx->b[i+2]);
	}
}

// Unit test
#ifdef __SHA256_UNITTEST__
	#define print_hash(x) printf("sha256 hash: "); for(int i=0;i<32;++i)printf("%02X", (x).b[i]);
	#define print_msg(x) printf("%s", ((x) ? "Pass":"Failed"))

int main(int argc, char **argv)
{
	SHA256 ctx;
	
	// ------------------ Stage 1: abc
	printf("------- Stage 1 : abc -------\n");
	BYTE abc[] = "abc";
	BYTE abcans[] = {0xBA, 0x78, 0x16, 0xBF, 0x8F, 0x01, 0xCF, 0xEA, 
					 0x41, 0x41, 0x40, 0xDE, 0x5D, 0xAE, 0x22, 0x23, 
					 0xB0, 0x03, 0x61, 0xA3, 0x96, 0x17, 0x7A, 0x9C, 
					 0xB4, 0x10, 0xFF, 0x61, 0xF2, 0x00, 0x15, 0xAD};
	size_t abclen = sizeof(abc) - 1;
	sha256(&ctx, abc, abclen);
	print_hash(ctx);
	printf("\nResult: ");
	print_msg(!memcmp(abcans, ctx.b, 32));
	printf("\n\n");
	
	// ------------------ Stage 2: len55
	printf("------ Stage 2 : len55 ------\n");
	BYTE len55[] = "1234567890123456789012345678901234567890123456789012345";
	BYTE len55ans[] = {0x03, 0xC3, 0xA7, 0x0E, 0x99, 0xED, 0x5E, 0xEC, 
					   0xCD, 0x80, 0xF7, 0x37, 0x71, 0xFC, 0xF1, 0xEC, 
					   0xE6, 0x43, 0xD9, 0x39, 0xD9, 0xEC, 0xC7, 0x6F, 
					   0x25, 0x54, 0x4B, 0x02, 0x33, 0xF7, 0x08, 0xE9};
	size_t len55len = sizeof(len55) - 1;
	sha256(&ctx, len55, len55len);
	print_hash(ctx);
	printf("\nResult: ");
	print_msg(!memcmp(len55ans, ctx.b, 32));
	printf("\n\n");
	
	// ------------------ Stage 3: len290
	printf("----- Stage 3 : len290 ------\n");
	BYTE len290[] = "ads;flkjas;dlkfjads;flkjads;flkafdlkjhfdalkjgadslfkjhadsjhfveroi"
					"uhwerpiuhwerptoiuywerptoiuywterypoihslgkjhdxzflgknbzsfdlkgjhsdfp"
					"gikjhwepgoiuhywertpiuywerptiuywrtoiuhwserlkjhsfdlgkjbsfd,nkmbxcv"
					".bkmnxflkjbnfdslgkjhsgpoiuhserpiuywerpituywetrpoiuhywerlkjbsfd,g"
					"nkbxsflkdjbsdflkjhsgfdluhsdgliuher";
	BYTE len290ans[] = {0xBD, 0xB5, 0xD4, 0xC1, 0xFB, 0x45, 0x1A, 0xD2, 
						0xFC, 0x8E, 0x62, 0x26, 0xF9, 0x5C, 0x6B, 0x58, 
						0x31, 0x53, 0x90, 0x1B, 0xE3, 0x74, 0xC2, 0x60, 
						0xC8, 0xA7, 0x46, 0x09, 0xC6, 0x89, 0x24, 0x60};
	size_t len290len = sizeof(len290) - 1;
	sha256(&ctx, len290, len290len);
	print_hash(ctx);
	printf("\nResult: ");
	print_msg(!memcmp(len290ans, ctx.b, 32));
	printf("\n\n");
	
	return 0;
}
#endif  //__SHA256_UNITTEST__

#ifdef __cplusplus
}
#endif  //__cplusplus

#undef _rotl
#undef _rotr
