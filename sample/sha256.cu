#include "sha256.h"
#include <string.h>

#include <cuda_runtime.h>

// ================== 通用巨集 ==================

#define CH(x,y,z)   (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z)  (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define ROTR(x,n)   (((x) >> (n)) | ((x) << (32u - (n))))
#define EP0(x)      (ROTR((x), 2) ^ ROTR((x),13) ^ ROTR((x),22))
#define EP1(x)      (ROTR((x), 6) ^ ROTR((x),11) ^ ROTR((x),25))
#define SIG0(x)     (ROTR((x), 7) ^ ROTR((x),18) ^ ((x) >> 3))
#define SIG1(x)     (ROTR((x),17) ^ ROTR((x),19) ^ ((x) >>10))

// ================== CPU 常數 ==================

static const WORD k_cpu[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

// ================== CPU 實作 ==================

static inline void _swap_cpu(BYTE *x, BYTE *y) {
    BYTE t = *x; *x = *y; *y = t;
}

extern "C"
void sha256_transform_cpu(SHA256 *ctx, const BYTE *msg)
{
    WORD w[64];

    for (WORD i = 0, j = 0; i < 16; ++i, j += 4) {
        w[i] = ((WORD)msg[j]   << 24)
             | ((WORD)msg[j+1] << 16)
             | ((WORD)msg[j+2] <<  8)
             | ((WORD)msg[j+3]);
    }

    for (WORD i = 16; i < 64; ++i) {
        WORD s0 = SIG0(w[i-15]);
        WORD s1 = SIG1(w[i-2]);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    WORD a = ctx->h[0];
    WORD b = ctx->h[1];
    WORD c = ctx->h[2];
    WORD d = ctx->h[3];
    WORD e = ctx->h[4];
    WORD f = ctx->h[5];
    WORD g = ctx->h[6];
    WORD h = ctx->h[7];

    for (WORD i = 0; i < 64; ++i) {
        WORD T1 = h + EP1(e) + CH(e,f,g) + k_cpu[i] + w[i];
        WORD T2 = EP0(a) + MAJ(a,b,c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
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

extern "C"
void sha256_midstate_cpu(const BYTE *block64, WORD midstate[8])
{
    SHA256 ctx;
    // 初始 IV（跟 sha256_cpu 一樣）
    ctx.h[0] = 0x6a09e667u;
    ctx.h[1] = 0xbb67ae85u;
    ctx.h[2] = 0x3c6ef372u;
    ctx.h[3] = 0xa54ff53au;
    ctx.h[4] = 0x510e527fu;
    ctx.h[5] = 0x9b05688cu;
    ctx.h[6] = 0x1f83d9abu;
    ctx.h[7] = 0x5be0cd19u;

    // 只對「第一個 512-bit block」做一次 transform，不做 padding
    sha256_transform_cpu(&ctx, block64);

    // 這就是 midstate
    for (int i = 0; i < 8; ++i)
        midstate[i] = ctx.h[i];
}


extern "C"
void sha256_cpu(SHA256 *ctx, const BYTE *msg, size_t len)
{
    ctx->h[0] = 0x6a09e667u;
    ctx->h[1] = 0xbb67ae85u;
    ctx->h[2] = 0x3c6ef372u;
    ctx->h[3] = 0xa54ff53au;
    ctx->h[4] = 0x510e527fu;
    ctx->h[5] = 0x9b05688cu;
    ctx->h[6] = 0x1f83d9abu;
    ctx->h[7] = 0x5be0cd19u;

    size_t i = 0;
    size_t full = len & ~((size_t)63);

    for (; i < full; i += 64)
        sha256_transform_cpu(ctx, msg + i);

    BYTE block[64] = {0};
    size_t rem = len - i;
    for (size_t j = 0; j < rem; ++j)
        block[j] = msg[i + j];

    block[rem++] = 0x80;

    if (rem > 56) {
        sha256_transform_cpu(ctx, block);
        memset(block, 0, 64);
    }

    unsigned long long bitlen = (unsigned long long)len * 8ull;
    block[63] = (BYTE)(bitlen      );
    block[62] = (BYTE)(bitlen >> 8 );
    block[61] = (BYTE)(bitlen >>16 );
    block[60] = (BYTE)(bitlen >>24 );
    block[59] = (BYTE)(bitlen >>32 );
    block[58] = (BYTE)(bitlen >>40 );
    block[57] = (BYTE)(bitlen >>48 );
    block[56] = (BYTE)(bitlen >>56 );

    sha256_transform_cpu(ctx, block);

    for (int j = 0; j < 32; j += 4) {
        _swap_cpu(&ctx->b[j],   &ctx->b[j+3]);
        _swap_cpu(&ctx->b[j+1], &ctx->b[j+2]);
    }
}

// ================== GPU 實作 ==================

__device__ __constant__ WORD k_gpu[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

__device__ __forceinline__ WORD rotr_gpu(WORD v, int s) {
#if __CUDA_ARCH__ >= 350
    return __funnelshift_r(v, v, s);
#else
    return (v >> s) | (v << (32 - s));
#endif
}

// ---- 通用 transform（給 sha256_gpu，用 w[16] 環形）----

__device__
void sha256_transform_gpu(SHA256 *ctx, const BYTE *msg)
{
    WORD w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15;

    // 讀前 16 個 word
#define LD_W(i, w) \
    w = ((WORD)msg[4*(i)+0] << 24) | \
        ((WORD)msg[4*(i)+1] << 16) | \
        ((WORD)msg[4*(i)+2] <<  8) | \
        ((WORD)msg[4*(i)+3]);

    LD_W(0, w0);  LD_W(1, w1);  LD_W(2, w2);  LD_W(3, w3);
    LD_W(4, w4);  LD_W(5, w5);  LD_W(6, w6);  LD_W(7, w7);
    LD_W(8, w8);  LD_W(9, w9);  LD_W(10,w10); LD_W(11,w11);
    LD_W(12,w12); LD_W(13,w13); LD_W(14,w14); LD_W(15,w15);
#undef LD_W

    WORD a = ctx->h[0];
    WORD b = ctx->h[1];
    WORD c = ctx->h[2];
    WORD d = ctx->h[3];
    WORD e = ctx->h[4];
    WORD f = ctx->h[5];
    WORD g = ctx->h[6];
    WORD h = ctx->h[7];

#define ROUND(W,K) do { \
    WORD T1 = h + EP1(e) + CH(e,f,g) + (K) + (W); \
    WORD T2 = EP0(a) + MAJ(a,b,c); \
    h = g; \
    g = f; \
    f = e; \
    e = d + T1; \
    d = c; \
    c = b; \
    b = a; \
    a = T1 + T2; \
} while(0)

#define SCHED() do { \
    WORD s0 = SIG0(w1); \
    WORD s1 = SIG1(w14); \
    WORD new_w = w0 + s0 + w9 + s1; \
    w0=w1; w1=w2; w2=w3; w3=w4; w4=w5; w5=w6; w6=w7; w7=w8; \
    w8=w9; w9=w10; w10=w11; w11=w12; w12=w13; w13=w14; w14=w15; w15=new_w; \
} while(0)

    // 前 16 round（直接用 w0..w15）
    ROUND(w0,  k_gpu[0]);  ROUND(w1,  k_gpu[1]);
    ROUND(w2,  k_gpu[2]);  ROUND(w3,  k_gpu[3]);
    ROUND(w4,  k_gpu[4]);  ROUND(w5,  k_gpu[5]);
    ROUND(w6,  k_gpu[6]);  ROUND(w7,  k_gpu[7]);
    ROUND(w8,  k_gpu[8]);  ROUND(w9,  k_gpu[9]);
    ROUND(w10, k_gpu[10]); ROUND(w11, k_gpu[11]);
    ROUND(w12, k_gpu[12]); ROUND(w13, k_gpu[13]);
    ROUND(w14, k_gpu[14]); ROUND(w15, k_gpu[15]);

    // 之後每一輪：先 SCHED() 更新環形，再用 w15
#pragma unroll
    for (int i = 16; i < 64; ++i) {
        SCHED();
        ROUND(w15, k_gpu[i]);
    }

#undef SCHED
#undef ROUND

    ctx->h[0] += a;
    ctx->h[1] += b;
    ctx->h[2] += c;
    ctx->h[3] += d;
    ctx->h[4] += e;
    ctx->h[5] += f;
    ctx->h[6] += g;
    ctx->h[7] += h;
}

// ---- 通用 sha256_gpu（給非 mining 用途；有 block[64]，會用 local memory，但不在熱路徑）----

__device__
void sha256_gpu(SHA256 *ctx, const BYTE *msg, size_t len)
{
    ctx->h[0] = 0x6a09e667u;
    ctx->h[1] = 0xbb67ae85u;
    ctx->h[2] = 0x3c6ef372u;
    ctx->h[3] = 0xa54ff53au;
    ctx->h[4] = 0x510e527fu;
    ctx->h[5] = 0x9b05688cu;
    ctx->h[6] = 0x1f83d9abu;
    ctx->h[7] = 0x5be0cd19u;

    size_t i = 0;
    while (i + 64 <= len) {
        sha256_transform_gpu(ctx, msg + i);
        i += 64;
    }

    BYTE block[64];
    size_t rem = len - i;
#pragma unroll
    for (size_t j = 0; j < rem; ++j)
        block[j] = msg[i + j];

    block[rem++] = 0x80;

    if (rem > 56) {
#pragma unroll
        for (size_t j = rem; j < 64; ++j) block[j] = 0;
        sha256_transform_gpu(ctx, block);
        rem = 0;
    }

#pragma unroll
    for (size_t j = rem; j < 56; ++j) block[j] = 0;

    unsigned long long bitlen = (unsigned long long)len * 8ull;
    block[63] = (BYTE)(bitlen      );
    block[62] = (BYTE)(bitlen >> 8 );
    block[61] = (BYTE)(bitlen >>16 );
    block[60] = (BYTE)(bitlen >>24 );
    block[59] = (BYTE)(bitlen >>32 );
    block[58] = (BYTE)(bitlen >>40 );
    block[57] = (BYTE)(bitlen >>48 );
    block[56] = (BYTE)(bitlen >>56 );

    sha256_transform_gpu(ctx, block);

#pragma unroll
    for (int j = 0; j < 32; j += 4) {
        BYTE t0 = ctx->b[j];
        BYTE t1 = ctx->b[j+1];
        ctx->b[j]   = ctx->b[j+3];
        ctx->b[j+1] = ctx->b[j+2];
        ctx->b[j+2] = t1;
        ctx->b[j+3] = t0;
    }
}

// ---- 通用 double_sha256_gpu ----

__device__
void double_sha256_gpu(SHA256 *out, const BYTE *bytes, size_t len)
{
    SHA256 tmp;
    sha256_gpu(&tmp, bytes, len);
    sha256_gpu(out, tmp.b, 32);
}

// ---- 專用：80-byte block header double SHA256，完全 scalar，無 64B 陣列 ----

__device__
void double_sha256_bitcoin_specialized(SHA256 *out, const BYTE *block80)
{
    const WORD IV0 = 0x6a09e667u;
    const WORD IV1 = 0xbb67ae85u;
    const WORD IV2 = 0x3c6ef372u;
    const WORD IV3 = 0xa54ff53au;
    const WORD IV4 = 0x510e527fu;
    const WORD IV5 = 0x9b05688cu;
    const WORD IV6 = 0x1f83d9abu;
    const WORD IV7 = 0x5be0cd19u;

    // -------- Round 1: first SHA over 80 bytes (two blocks) --------

    // Block 0: 前 64 bytes
    WORD w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15;

#define LD_W(i, w) \
    w = ((WORD)block80[4*(i)+0] << 24) | \
        ((WORD)block80[4*(i)+1] << 16) | \
        ((WORD)block80[4*(i)+2] <<  8) | \
        ((WORD)block80[4*(i)+3]);

    LD_W(0, w0);  LD_W(1, w1);  LD_W(2, w2);  LD_W(3, w3);
    LD_W(4, w4);  LD_W(5, w5);  LD_W(6, w6);  LD_W(7, w7);
    LD_W(8, w8);  LD_W(9, w9);  LD_W(10,w10); LD_W(11,w11);
    LD_W(12,w12); LD_W(13,w13); LD_W(14,w14); LD_W(15,w15);
#undef LD_W

    WORD a = IV0, b = IV1, c = IV2, d = IV3;
    WORD e = IV4, f = IV5, g = IV6, h = IV7;

#define ROUND(W,K) do{ \
    WORD T1 = h + EP1(e) + CH(e,f,g) + (K) + (W); \
    WORD T2 = EP0(a) + MAJ(a,b,c); \
    h = g; \
    g = f; \
    f = e; \
    e = d + T1; \
    d = c; \
    c = b; \
    b = a; \
    a = T1 + T2; \
}while(0)

#define SCHED() do{ \
    WORD s0 = SIG0(w1); \
    WORD s1 = SIG1(w14); \
    WORD new_w = w0 + s0 + w9 + s1; \
    w0=w1; w1=w2; w2=w3; w3=w4; w4=w5; w5=w6; w6=w7; w7=w8; \
    w8=w9; w9=w10; w10=w11; w11=w12; w12=w13; w13=w14; w14=w15; w15=new_w; \
}while(0)

    // Block 0: 16 rounds
    ROUND(w0,  k_gpu[0]);  ROUND(w1,  k_gpu[1]);
    ROUND(w2,  k_gpu[2]);  ROUND(w3,  k_gpu[3]);
    ROUND(w4,  k_gpu[4]);  ROUND(w5,  k_gpu[5]);
    ROUND(w6,  k_gpu[6]);  ROUND(w7,  k_gpu[7]);
    ROUND(w8,  k_gpu[8]);  ROUND(w9,  k_gpu[9]);
    ROUND(w10, k_gpu[10]); ROUND(w11, k_gpu[11]);
    ROUND(w12, k_gpu[12]); ROUND(w13, k_gpu[13]);
    ROUND(w14, k_gpu[14]); ROUND(w15, k_gpu[15]);

    // Block 0: 擴展 rounds 16..63
#pragma unroll
    for (int i = 16; i < 64; ++i) {
        SCHED();
        ROUND(w15, k_gpu[i]);
    }

    // 累加成 Block0 後的 state
    WORD r0 = IV0 + a;
    WORD r1 = IV1 + b;
    WORD r2 = IV2 + c;
    WORD r3 = IV3 + d;
    WORD r4 = IV4 + e;
    WORD r5 = IV5 + f;
    WORD r6 = IV6 + g;
    WORD r7 = IV7 + h;

    // Block 1: 後 16 bytes + padding (總長 80 bytes)

    // 重設 a..h = r0..r7
    a = r0; b = r1; c = r2; d = r3;
    e = r4; f = r5; g = r6; h = r7;

    // w0..w3 = block80[64..79]
    w0 = ((WORD)block80[64] << 24) |
         ((WORD)block80[65] << 16) |
         ((WORD)block80[66] <<  8) |
         ((WORD)block80[67]);
    w1 = ((WORD)block80[68] << 24) |
         ((WORD)block80[69] << 16) |
         ((WORD)block80[70] <<  8) |
         ((WORD)block80[71]);
    w2 = ((WORD)block80[72] << 24) |
         ((WORD)block80[73] << 16) |
         ((WORD)block80[74] <<  8) |
         ((WORD)block80[75]);
    w3 = ((WORD)block80[76] << 24) |
         ((WORD)block80[77] << 16) |
         ((WORD)block80[78] <<  8) |
         ((WORD)block80[79]);

    w4  = 0x80000000u;
    w5  = 0u; w6  = 0u; w7  = 0u;
    w8  = 0u; w9  = 0u; w10 = 0u; w11 = 0u;
    w12 = 0u; w13 = 0u;
    w14 = 0u;
    w15 = 0x00000280u; // 80 * 8

    // 16 rounds
    ROUND(w0,  k_gpu[0]);  ROUND(w1,  k_gpu[1]);
    ROUND(w2,  k_gpu[2]);  ROUND(w3,  k_gpu[3]);
    ROUND(w4,  k_gpu[4]);  ROUND(w5,  k_gpu[5]);
    ROUND(w6,  k_gpu[6]);  ROUND(w7,  k_gpu[7]);
    ROUND(w8,  k_gpu[8]);  ROUND(w9,  k_gpu[9]);
    ROUND(w10, k_gpu[10]); ROUND(w11, k_gpu[11]);
    ROUND(w12, k_gpu[12]); ROUND(w13, k_gpu[13]);
    ROUND(w14, k_gpu[14]); ROUND(w15, k_gpu[15]);

    // 擴展 rounds 16..63
#pragma unroll
    for (int i = 16; i < 64; ++i) {
        SCHED();
        ROUND(w15, k_gpu[i]);
    }

    // 得到 first SHA 結果
    WORD h0 = r0 + a;
    WORD h1 = r1 + b;
    WORD h2 = r2 + c;
    WORD h3 = r3 + d;
    WORD h4 = r4 + e;
    WORD h5 = r5 + f;
    WORD h6 = r6 + g;
    WORD h7 = r7 + h;

    // -------- Round 2: SHA256(32-byte digest) --------

    a = IV0; b = IV1; c = IV2; d = IV3;
    e = IV4; f = IV5; g = IV6; h = IV7;

    // 前 8 word = h0..h7
    w0 = h0; w1 = h1; w2 = h2; w3 = h3;
    w4 = h4; w5 = h5; w6 = h6; w7 = h7;

    // padding：1 bit 後全 0，長度 256 bits
    w8  = 0x80000000u;
    w9  = 0u; w10 = 0u; w11 = 0u;
    w12 = 0u; w13 = 0u;
    w14 = 0u;
    w15 = 0x00000100u; // 32 * 8

    // 16 rounds
    ROUND(w0,  k_gpu[0]);  ROUND(w1,  k_gpu[1]);
    ROUND(w2,  k_gpu[2]);  ROUND(w3,  k_gpu[3]);
    ROUND(w4,  k_gpu[4]);  ROUND(w5,  k_gpu[5]);
    ROUND(w6,  k_gpu[6]);  ROUND(w7,  k_gpu[7]);
    ROUND(w8,  k_gpu[8]);  ROUND(w9,  k_gpu[9]);
    ROUND(w10, k_gpu[10]); ROUND(w11, k_gpu[11]);
    ROUND(w12, k_gpu[12]); ROUND(w13, k_gpu[13]);
    ROUND(w14, k_gpu[14]); ROUND(w15, k_gpu[15]);

#pragma unroll
    for (int i = 16; i < 64; ++i) {
        SCHED();
        ROUND(w15, k_gpu[i]);
    }

    // 加回 IV 得到 final hash
    h0 = IV0 + a;
    h1 = IV1 + b;
    h2 = IV2 + c;
    h3 = IV3 + d;
    h4 = IV4 + e;
    h5 = IV5 + f;
    h6 = IV6 + g;
    h7 = IV7 + h;

#undef ROUND
#undef SCHED

    // 寫成 big-endian bytes
    WORD v;
    v = h0; out->b[ 0] = (BYTE)(v>>24); out->b[ 1] = (BYTE)(v>>16);
             out->b[ 2] = (BYTE)(v>> 8); out->b[ 3] = (BYTE)(v    );
    v = h1; out->b[ 4] = (BYTE)(v>>24); out->b[ 5] = (BYTE)(v>>16);
             out->b[ 6] = (BYTE)(v>> 8); out->b[ 7] = (BYTE)(v    );
    v = h2; out->b[ 8] = (BYTE)(v>>24); out->b[ 9] = (BYTE)(v>>16);
             out->b[10] = (BYTE)(v>> 8); out->b[11] = (BYTE)(v    );
    v = h3; out->b[12] = (BYTE)(v>>24); out->b[13] = (BYTE)(v>>16);
             out->b[14] = (BYTE)(v>> 8); out->b[15] = (BYTE)(v    );
    v = h4; out->b[16] = (BYTE)(v>>24); out->b[17] = (BYTE)(v>>16);
             out->b[18] = (BYTE)(v>> 8); out->b[19] = (BYTE)(v    );
    v = h5; out->b[20] = (BYTE)(v>>24); out->b[21] = (BYTE)(v>>16);
             out->b[22] = (BYTE)(v>> 8); out->b[23] = (BYTE)(v    );
    v = h6; out->b[24] = (BYTE)(v>>24); out->b[25] = (BYTE)(v>>16);
             out->b[26] = (BYTE)(v>> 8); out->b[27] = (BYTE)(v    );
    v = h7; out->b[28] = (BYTE)(v>>24); out->b[29] = (BYTE)(v>>16);
             out->b[30] = (BYTE)(v>> 8); out->b[31] = (BYTE)(v    );
}

__device__
void double_sha256_from_midstate(SHA256 *out,
                                 const WORD midstate[8],
                                 const BYTE *block80)
{
    // 初始 IV
    const WORD IV0 = 0x6a09e667u;
    const WORD IV1 = 0xbb67ae85u;
    const WORD IV2 = 0x3c6ef372u;
    const WORD IV3 = 0xa54ff53au;
    const WORD IV4 = 0x510e527fu;
    const WORD IV5 = 0x9b05688cu;
    const WORD IV6 = 0x1f83d9abu;
    const WORD IV7 = 0x5be0cd19u;

    // ========= 第一輪：延續 midstate，處理 Block1 =========
    // Block1 layout (big-endian words):
    // w0  = merkle_root[28..31]
    // w1  = ntime
    // w2  = nbits
    // w3  = nonce
    // w4  = 0x80000000
    // w5..w14 = 0
    // w15 = 80 * 8 = 0x00000280

    WORD w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;

    // 從 block80[64..79] 抓最後 16 bytes（已含當前 nonce）
    w0 = ((WORD)block80[64] << 24) |
         ((WORD)block80[65] << 16) |
         ((WORD)block80[66] <<  8) |
         ((WORD)block80[67]);

    w1 = ((WORD)block80[68] << 24) |
         ((WORD)block80[69] << 16) |
         ((WORD)block80[70] <<  8) |
         ((WORD)block80[71]);

    w2 = ((WORD)block80[72] << 24) |
         ((WORD)block80[73] << 16) |
         ((WORD)block80[74] <<  8) |
         ((WORD)block80[75]);

    w3 = ((WORD)block80[76] << 24) |
         ((WORD)block80[77] << 16) |
         ((WORD)block80[78] <<  8) |
         ((WORD)block80[79]);

    w4  = 0x80000000u;
    w5  = 0u; w6  = 0u; w7  = 0u;
    w8  = 0u; w9  = 0u; w10 = 0u; w11 = 0u;
    w12 = 0u; w13 = 0u; w14 = 0u;
    w15 = 0x00000280u; // 80 * 8

    // 起始 state = midstate
    WORD a = midstate[0];
    WORD b = midstate[1];
    WORD c = midstate[2];
    WORD d = midstate[3];
    WORD e = midstate[4];
    WORD f = midstate[5];
    WORD g = midstate[6];
    WORD h = midstate[7];

#define ROUND(W,K) do{ \
    WORD T1 = h + EP1(e) + CH(e,f,g) + (K) + (W); \
    WORD T2 = EP0(a) + MAJ(a,b,c); \
    h = g; \
    g = f; \
    f = e; \
    e = d + T1; \
    d = c; \
    c = b; \
    b = a; \
    a = T1 + T2; \
}while(0)

#define SCHED() do{ \
    WORD s0 = SIG0(w1); \
    WORD s1 = SIG1(w14); \
    WORD new_w = w0 + s0 + w9 + s1; \
    w0=w1; w1=w2; w2=w3; w3=w4; w4=w5; w5=w6; w6=w7; w7=w8; \
    w8=w9; w9=w10; w10=w11; w11=w12; w12=w13; w13=w14; w14=w15; w15=new_w; \
}while(0)

    // 16 rounds with initial w0..w15
    ROUND(w0,  k_gpu[0]);  ROUND(w1,  k_gpu[1]);
    ROUND(w2,  k_gpu[2]);  ROUND(w3,  k_gpu[3]);
    ROUND(w4,  k_gpu[4]);  ROUND(w5,  k_gpu[5]);
    ROUND(w6,  k_gpu[6]);  ROUND(w7,  k_gpu[7]);
    ROUND(w8,  k_gpu[8]);  ROUND(w9,  k_gpu[9]);
    ROUND(w10, k_gpu[10]); ROUND(w11, k_gpu[11]);
    ROUND(w12, k_gpu[12]); ROUND(w13, k_gpu[13]);
    ROUND(w14, k_gpu[14]); ROUND(w15, k_gpu[15]);

    // schedule + rounds 16..63
#pragma unroll
    for (int i = 16; i < 64; ++i) {
        SCHED();
        ROUND(w15, k_gpu[i]);
    }

    // 第一輪 SHA 的 digest（32 bytes, big-endian words）
    WORD H0 = midstate[0] + a;
    WORD H1 = midstate[1] + b;
    WORD H2 = midstate[2] + c;
    WORD H3 = midstate[3] + d;
    WORD H4 = midstate[4] + e;
    WORD H5 = midstate[5] + f;
    WORD H6 = midstate[6] + g;
    WORD H7 = midstate[7] + h;

    // ========= 第二輪：SHA256(32-byte digest) =========
    a = IV0; b = IV1; c = IV2; d = IV3;
    e = IV4; f = IV5; g = IV6; h = IV7;

    w0 = H0; w1 = H1; w2 = H2; w3 = H3;
    w4 = H4; w5 = H5; w6 = H6; w7 = H7;
    w8  = 0x80000000u;
    w9  = 0u; w10 = 0u; w11 = 0u;
    w12 = 0u; w13 = 0u;
    w14 = 0u;
    w15 = 0x00000100u; // 32 * 8

    // 16 rounds
    ROUND(w0,  k_gpu[0]);  ROUND(w1,  k_gpu[1]);
    ROUND(w2,  k_gpu[2]);  ROUND(w3,  k_gpu[3]);
    ROUND(w4,  k_gpu[4]);  ROUND(w5,  k_gpu[5]);
    ROUND(w6,  k_gpu[6]);  ROUND(w7,  k_gpu[7]);
    ROUND(w8,  k_gpu[8]);  ROUND(w9,  k_gpu[9]);
    ROUND(w10, k_gpu[10]); ROUND(w11, k_gpu[11]);
    ROUND(w12, k_gpu[12]); ROUND(w13, k_gpu[13]);
    ROUND(w14, k_gpu[14]); ROUND(w15, k_gpu[15]);

#pragma unroll
    for (int i = 16; i < 64; ++i) {
        SCHED();
        ROUND(w15, k_gpu[i]);
    }

#undef SCHED
#undef ROUND

    H0 = IV0 + a;
    H1 = IV1 + b;
    H2 = IV2 + c;
    H3 = IV3 + d;
    H4 = IV4 + e;
    H5 = IV5 + f;
    H6 = IV6 + g;
    H7 = IV7 + h;

    // output big-endian bytes
    WORD v;
    v = H0; out->b[ 0] = (BYTE)(v>>24); out->b[ 1] = (BYTE)(v>>16);
             out->b[ 2] = (BYTE)(v>> 8); out->b[ 3] = (BYTE)(v    );
    v = H1; out->b[ 4] = (BYTE)(v>>24); out->b[ 5] = (BYTE)(v>>16);
             out->b[ 6] = (BYTE)(v>> 8); out->b[ 7] = (BYTE)(v    );
    v = H2; out->b[ 8] = (BYTE)(v>>24); out->b[ 9] = (BYTE)(v>>16);
             out->b[10] = (BYTE)(v>> 8); out->b[11] = (BYTE)(v    );
    v = H3; out->b[12] = (BYTE)(v>>24); out->b[13] = (BYTE)(v>>16);
             out->b[14] = (BYTE)(v>> 8); out->b[15] = (BYTE)(v    );
    v = H4; out->b[16] = (BYTE)(v>>24); out->b[17] = (BYTE)(v>>16);
             out->b[18] = (BYTE)(v>> 8); out->b[19] = (BYTE)(v    );
    v = H5; out->b[20] = (BYTE)(v>>24); out->b[21] = (BYTE)(v>>16);
             out->b[22] = (BYTE)(v>> 8); out->b[23] = (BYTE)(v    );
    v = H6; out->b[24] = (BYTE)(v>>24); out->b[25] = (BYTE)(v>>16);
             out->b[26] = (BYTE)(v>> 8); out->b[27] = (BYTE)(v    );
    v = H7; out->b[28] = (BYTE)(v>>24); out->b[29] = (BYTE)(v>>16);
             out->b[30] = (BYTE)(v>> 8); out->b[31] = (BYTE)(v    );
}

