/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
#include "runtime.h"
#include "util.h"
#include "model.h"
#include "kernel/bmpmm.h"
#include "tokenizer.h"
#include <string.h>

#include <riscv_vector.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
#include "win.h"
#else
#include <unistd.h>
#endif

#define DEBUG_LEVEL 0

#define DEBUG_PRINT(level, fmt, ...)    \
    do                                  \
    {                                   \
        if (DEBUG_LEVEL >= level)       \
            printf(fmt, ##__VA_ARGS__); \
    } while (0)

#define MAX_TOKEN_LENGTH 7
#define VOCAB_SIZE 512
#define DIM 64
#define HIDDEN_DIM 172
#define SEQ_LEN 512
#define N_HEADS 8
#define N_LAYERS 5
#define N_KV_HEADS 4
#define KV_DIM DIM *N_KV_HEADS / N_HEADS

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

typedef struct
{
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
} Config;

typedef struct
{
    int8_t *q; // quantized values
    float *s;  // scaling factors
} QuantizedTensor;

typedef struct
{
    // token embedding table
    QuantizedTensor *q_tokens;    // (vocab_size, dim)
    float *token_embedding_table; // same, but dequantized

    // weights for rmsnorms
    float *rms_att_weight; // (layer, dim) rmsnorm weights
    float *rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct
{
    // current wave of activations
    float *x;           // activation at current time stamp (dim,)
    float *xb;          // same, but inside a residual branch (dim,)
    float *xb2;         // an additional buffer just for convenience (dim,)
    float *hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q;           // query (dim,)
    float *k;           // key (dim,)
    float *v;           // value (dim,)
    float *att;         // buffer for scores/attention values (n_heads, seq_len)
    float *logits;      // output logits
    // kv cache
    float *key_cache;   // (layer, seq_len, dim)
    float *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct
{
    float x[DIM];
    float xb[DIM];
    float xb2[DIM];
    float hb[HIDDEN_DIM];
    float hb2[HIDDEN_DIM];
    int8_t xq_q[DIM];
    float xq_s[DIM];
    int8_t hq_q[HIDDEN_DIM];
    float hq_s[HIDDEN_DIM];
    float q[DIM];
    float k[KV_DIM];
    float v[KV_DIM];
    float att[N_HEADS * SEQ_LEN];
    float logits[VOCAB_SIZE];
    float key_cache[N_LAYERS * SEQ_LEN * KV_DIM];
    float value_cache[N_LAYERS * SEQ_LEN * KV_DIM];
} StaticRunState;

typedef struct
{
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;           // file descriptor for memory mapping
    float *data;      // memory mapped data pointer
    size_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, StaticRunState *buf)
{
    s->x = buf->x;
    s->xb = buf->xb;
    s->xb2 = buf->xb2;
    s->hb = buf->hb;
    s->hb2 = buf->hb2;

    s->xq.q = buf->xq_q;
    s->xq.s = buf->xq_s;
    s->hq.q = buf->hq_q;
    s->hq.s = buf->hq_s;

    s->q = buf->q;
    s->k = buf->k;
    s->v = buf->v;

    s->att = buf->att;
    s->logits = buf->logits;
    s->key_cache = buf->key_cache;
    s->value_cache = buf->value_cache;
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n)
{
    GS = n;
    for (int i = 0; i < n; i++)
    {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float *x, int n)
{
    GS = n;
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++)
    {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++)
        {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
            {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++)
        {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t)round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

// /* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, QuantizedTensor *res, int n, int size_each)
{
    GS = n;
    void *p = *ptr;
    for (int i = 0; i < n; i++)
    {
        /* map quantized int8 values*/
        res[i].q = (int8_t *)p;
        p = (int8_t *)p + size_each;
        /* map scale factors */
        res[i].s = (float *)p;
        p = (float *)p + size_each / GS;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

static QuantizedTensor q_tokens[1] __attribute__((aligned(4 * NR_LANES), section(".l2"))); // global quantized token embedding table
static QuantizedTensor wq[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor wk[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor wv[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor wo[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor w1[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor w2[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor w3[N_LAYERS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static QuantizedTensor wcls[1] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static float token_embedding_table_buffer[VOCAB_SIZE * DIM] __attribute__((aligned(4 * NR_LANES), section(".l2")));

void memory_map_weights(TransformerWeights *w, void *ptr, uint8_t shared_classifier)
{
    int head_size = DIM / N_HEADS;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float *fptr = (float *)ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += N_LAYERS * DIM;
    w->rms_ffn_weight = fptr;
    fptr += N_LAYERS * DIM;
    w->rms_final_weight = fptr;
    fptr += DIM;

    // now read all the quantized weights
    ptr = (void *)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, q_tokens, 1, VOCAB_SIZE * DIM);
    printf("q_tokens initialized\n");
    // dequantize token embedding table
    // w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    w->token_embedding_table = token_embedding_table_buffer;
    dequantize(w->q_tokens, w->token_embedding_table, VOCAB_SIZE * DIM);

    w->wq = init_quantized_tensors(&ptr, wq, N_LAYERS, DIM * (N_HEADS * head_size));
    printf("wq initialized\n");
    w->wk = init_quantized_tensors(&ptr, wk, N_LAYERS, DIM * (N_KV_HEADS * head_size));
    printf("wk initialized\n");
    w->wv = init_quantized_tensors(&ptr, wv, N_LAYERS, DIM * (N_KV_HEADS * head_size));
    printf("wv initialized\n");
    w->wo = init_quantized_tensors(&ptr, wo, N_LAYERS, (N_HEADS * head_size) * DIM);
    printf("wo initialized\n");

    w->w1 = init_quantized_tensors(&ptr, w1, N_LAYERS, DIM * HIDDEN_DIM);
    printf("w1 initialized\n");
    w->w2 = init_quantized_tensors(&ptr, w2, N_LAYERS, HIDDEN_DIM * DIM);
    printf("w2 initialized\n");
    w->w3 = init_quantized_tensors(&ptr, w3, N_LAYERS, DIM * HIDDEN_DIM);
    printf("w3 initialized\n");

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, wcls, 1, DIM * VOCAB_SIZE);
}

void read_checkpoint(Config *config, TransformerWeights *weights, float **data, size_t *file_size)
{
    // 设置文件大小
    *file_size = model_bin_len;

    *data = (float *)model_bin;

    // 校验
    uint32_t magic_number = 0;
    memcpy(&magic_number, model_bin, sizeof(uint32_t));
    if (magic_number != 0x616b3432)
    {
        printf("Bad magic number\n");
    }

    // 校验版本号
    int version = 0;
    memcpy(&version, model_bin + sizeof(uint32_t), sizeof(int));
    if (version != 2)
    {
        printf("Bad version\n");
    }

    // 解析 Config
    memcpy(config, model_bin + sizeof(uint32_t) + sizeof(int), sizeof(Config));

    // 读取 shared_classifier 和 group_size
    uint8_t shared_classifier = 0;
    memcpy(&shared_classifier, model_bin + sizeof(uint32_t) + sizeof(int) + sizeof(Config), sizeof(uint8_t));

    int group_size = 0;
    memcpy(&group_size, model_bin + sizeof(uint32_t) + sizeof(int) + sizeof(Config) + sizeof(uint8_t), sizeof(int));
    GS = group_size;

    // 权重映射：跳过前 256 字节
    void *weights_ptr = (void *)(model_bin + 256);
    printf("Begin memory_map_weights:\n");
    memory_map_weights(weights, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t)
{
    printf("┌────────────────────────────────────────┐\n");
    printf("│  Begin building Transformer model      │\n");
    printf("└────────────────────────────────────────┘\n");

    read_checkpoint(&t->config, &t->weights, &t->data, &t->file_size);

    printf("┌──────────────────────────────┐\n");
    printf("│ model config:                │\n");
    printf("├──────────────────────────────┤\n");
    printf("│ dim: %-16d        │\n", t->config.dim);
    printf("│ hidden_dim: %-14d   │\n", t->config.hidden_dim);
    printf("│ n_layers: %-19d│\n", t->config.n_layers);
    printf("│ n_heads: %-16d    │\n", t->config.n_heads);
    printf("│ n_kv_heads: %-17d│\n", t->config.n_kv_heads);
    printf("│ vocab_size: %-17d│\n", t->config.vocab_size);
    printf("│ seq_len: %-17d   │\n", t->config.seq_len);
    printf("└──────────────────────────────┘\n");

    StaticRunState run_state_buffer;
    malloc_run_state(&t->state, &run_state_buffer);
    printf("✓ RunState mallocated\n");
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size)
{
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
    {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++)
    {
        asm volatile("mpcfg 9\n\t");
        float a = weight[j];
        asm volatile("mpcfg 10\n\t");
        float b = ss * x[j];
        asm volatile("mpcfg 11\n\t");
        float tmp = a * b;
        asm volatile("mpcfg 12\n\t");
        o[j] = tmp;
    }
}

void softmax(float *x, int size)
{
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++)
    {
        if (x[i] > max_val)
        {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++)
    {
        x[i] /= sum;
    }
}

void transpose(const int8_t *A, int8_t *B, int M, int N)
{
    for (int col = 0; col < N; col++)
    {
        int i = 0;
        while (i < M)
        {
            size_t vl;
            const int8_t *src = A + i * N + col;
            int8_t *dst = B + col * M + i;
            __asm__ volatile("vsetvli %[vl], %[remain], e8, m1, ta, ma" 
                : [vl] "=&r"(vl)
                : [remain] "r"(M - i)
                : "memory");
            __asm__ volatile("vlse8.v v0, (%[src]), %[stride]"
                :
                : [src] "r"(src),
                  [stride] "r"((ptrdiff_t)(N))
                : "memory", "v0");
            __asm__ volatile("vse8.v v0, (%[dst])"
                :
                : [dst] "r"(dst)
                : "memory", "v0");
            i += vl;
        }
    }
}

#define MAX_PADDED_K 480
#define MAX_CHUNKS_PER_ROW (MAX_PADDED_K / 8)

// 静态分配缓冲区
static int8_t padded_row_buf[488]__attribute__((aligned(4 * NR_LANES), section(".l2")));               // 最大需要 488 字节
static uint64_t block_matrix[16][MAX_CHUNKS_PER_ROW]__attribute__((aligned(4 * NR_LANES), section(".l2")));

/**
 * pack_activation - 打包激活矩阵以适配后续向量化计算
 * @array: 输入矩阵 (M x K)
 * @M: 行数
 * @K: 列数
 * @out: 输出 buffer，用于存储打包后的数据
 * @padded_row_buf: 外部提供的临时 buffer，大小 >= 488 字节
 * @block_matrix: 外部提供的一块 buffer，大小 = 16 * chunks_per_row * sizeof(uint64_t)
 *
 * 返回值：写入 out 的字节数
 */
size_t pack_activation(
    const int8_t* array,
    int M,
    int K,
    int8_t* out
) {
    int tm = (M + 15) / 16;        // number of tiles in m dimension
    int tk = (K > 480) ? ((K + 479) / 480) : 1;

    int8_t* out_start = out;

    for (int i = 0; i < tm; ++i) {
        int m_start = i * 16;
        int m_valid = M - m_start;
        int m_tile = (m_valid < 0) ? 0 : ((m_valid > 16) ? 16 : m_valid);

        for (int jj = 0; jj < tk; ++jj) {
            int k_start = jj * 480;
            int k_end = k_start + 480;
            if (k_end > K) k_end = K;
            int k_tile = k_end - k_start;

            // round up to multiple of 8
            int padded_k = (k_tile + 7) & (~7);  // 向上对齐到 8 的倍数
            int chunks_per_row = padded_k / 8;

            // 指向 block_matrix 缓冲区
            uint64_t (*block)[chunks_per_row] = (uint64_t (*)[chunks_per_row])block_matrix;

            for (int r = 0; r < 16; ++r) {
                int src_row = m_start + r;
                const int8_t* src_data;

                if (src_row >= M || r >= m_tile) {
                    // Zero pad the row
                    static int8_t zero_pad[488];
                    src_data = zero_pad;
                } else {
                    src_data = &array[src_row * K + k_start];
                }

                // Copy and pad to padded_row
                memcpy(padded_row_buf, src_data, k_tile);
                memset(padded_row_buf + k_tile, 0, padded_k - k_tile);

                // Split into 8-byte chunks, reverse and pack
                for (int c = 0; c < chunks_per_row; ++c) {
                    const int8_t* chunk = padded_row_buf + c * 8;
                    uint64_t packed = 0;

                    for (int b = 0; b < 8; ++b) {
                        packed <<= 8;
                        packed |= (uint8_t)(chunk[7 - b]);  // Reverse order
                    }

                    block[r][c] = packed;
                }
            }

            // Now transpose block_matrix (16 x chunks_per_row) -> (chunks_per_row x 16)
            for (int c = 0; c < chunks_per_row; ++c) {
                for (int r = 0; r < 16; ++r) {
                    uint64_t val = block[r][c];
                    memcpy(out, &val, 8);
                    out += 8;
                }
            }
        }
    }

    return out - out_start;
}
static int8_t transposed_buffer[512*172] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int8_t packed_activation[512] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int16_t matmul_out_buffer[512] __attribute__((aligned(4 * NR_LANES), section(".l2")));
void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    // printf("matmul shape: W(%d, %d) @ x(%d,) -> xout(%d,)\n", d, n, n, d);
    int8_t *transposed_weight = transposed_buffer;
    transpose(w->q, transposed_weight, n, d);
    pack_activation(x->q, 1, n, packed_activation);
    binary_mixed_matmul(matmul_out_buffer, packed_activation, transposed_weight, 1, n, d);
    int i = 0;
    while (i < d) {
        size_t vl;
        __asm__ volatile (
            // 设置向量长度
            "vsetvli    %[vl], %[remain], e16, m1, ta, ma          \n"
            // 加载 int16 向量到 v0
            "vle16.v    v0, (%[in])                        \n"
            // 扩展 int16 到 int32（有符号扩展）
            "vsext.vf2  v1, v0                             \n"
            // 转换 int32 -> float32
            "vfcvt.f.x.v v2, v1                            \n"
            // 标量广播 float -> 向量
            "vfmv.v.f   v3, %[scale]                       \n"
            // 浮点乘法
            "vfmul.vv   v4, v2, v3                         \n"
            // 存储 float32 结果
            "vse32.v    v4, (%[out])                       \n"

            : [vl] "=&r" (vl)  // 输出：向量长度寄存器
            : [in] "r" (matmul_out_buffer + i),       // 输入矩阵指针
              [out] "r" (xout + i),     // 输出指针
              [scale] "f" (x->s[0]+w->s[0]),      // 标量因子
              [remain] "r" (d - i)   // 剩余处理元素
            : "v0", "v1", "v2", "v3", "v4", "memory"
        );
        i += vl;
    }
    // printf("%d\n", i);
    // vector_int8_matmul(xout, x->q, w->q, 1, n, d);
}

// void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
//     // W (d,n) @ x (n,) -> xout (d,)
//     // by far the most amount of time is spent inside this little function
//     // inputs to this function are both quantized
//     // printf matmul shape
//     // printf("matmul shape: W(%d, %d) @ x(%d,) -> xout(%d,)\n", d, n, n, d);

//     int i;
//     #pragma omp parallel for private(i)
//     for (i = 0; i < d; i++) {

//         float val = 0.0f;
//         int32_t ival = 0;
//         int in = i * n;

//         // do the matmul in groups of GS
//         int j;
//         for (j = 0; j <= n - GS; j += GS) {
//             for (int k = 0; k < GS; k++) {
//                 ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
//             }
//             val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
//             ival = 0;
//         }

//         xout[i] = val;
//     }
// }

float *forward(Transformer *transformer, int token, int pos)
{
    DEBUG_PRINT(2, "┌── Begin forward pass (pos=%d, token=%d)\n", pos, token);
    // a few convenience variables
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = DIM;
    int kv_dim = (DIM * N_KV_HEADS) / N_HEADS;
    int kv_mul = N_HEADS / N_KV_HEADS; // integer multiplier of the kv sharing in multiquery
    int hidden_dim = HIDDEN_DIM;
    int head_size = dim / N_HEADS;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

    // forward all the layers
    for (int l = 0; l < N_LAYERS; l++)
    {
        start_timer();
        DEBUG_PRINT(2, "├─ %d layer computation begin\n", l + 1);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        // qkv matmuls
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
        DEBUG_PRINT(3, "│  ✓ Q/K/V computation finished\n");
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i += 2)
        {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++)
            {
                float *vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float *key_cache_row = s->key_cache + loff + pos * kv_dim;
        float *value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        DEBUG_PRINT(3, "│  ✓ K/V cache update finished\n");
        // multihead attention. iterate over all heads
        int h;
        for (h = 0; h < N_HEADS; h++)
        {
            // get the query vector for this head
            float *q = s->q + h * head_size;
            // attention scores for this head
            float *att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++)
            {
                // get the key vector for this head and at this timestep
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++)
                {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++)
            {
                // get the value vector for this head and at this timestep
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++)
                {
                    xb[i] += a * v[i];
                }
            }
        }
        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
        DEBUG_PRINT(3, "│  ✓ MHA output computaiton finished\n");
        // residual connection back into x
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb2[i];
        }
        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);
        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++)
        {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);
        DEBUG_PRINT(3, "│  ✓ FFN output matmul finished\n");
        // residual connection
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
        DEBUG_PRINT(2, "├─ %d layer computation finished\n", l + 1);
        stop_timer();
        int64_t runtime = get_timer();
        printf("layer%d spent cycles: %ld\n", l, runtime);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, VOCAB_SIZE);
    DEBUG_PRINT(2, "└── forward pass finished\n");
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char (*vocab)[MAX_TOKEN_LENGTH];
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

static TokenIndex sorted_vocab[VOCAB_SIZE] __attribute__((aligned(4 * NR_LANES), section(".l2")));
void build_tokenizer(Tokenizer *t)
{
    t->vocab_size = VOCAB_SIZE;
    t->vocab = tokenizer_vocab;
    t->max_token_length = MAX_TOKEN_LENGTH;
    t->vocab_scores = tokenizer_vocab_scores;
    t->sorted_vocab = sorted_vocab;

    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
}

// 判断是否是有效十六进制字符
int is_hex_digit(char c)
{
    return (c >= '0' && c <= '9') ||
           (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

// 将 '0'-'9', 'a'-'f', 'A'-'F' 转换为对应数值
unsigned char hex_char_to_val(char c)
{
    if (c >= '0' && c <= '9')
        return c - '0';
    if (c >= 'a' && c <= 'f')
        return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
        return c - 'A' + 10;
    return 0; // 默认返回 0（输入已检查过有效性）
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];

    // 去除 BOS (1) 后的空格
    if (prev_token == 1 && piece[0] == ' ')
    {
        piece++; // 指向非空格字符
    }

    // 判断是否是形如 "<0xAB>" 的原始字节表示（6 个字符）
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' &&
        is_hex_digit(piece[3]) && is_hex_digit(piece[4]) && piece[5] == '>')
    {
        // 手动解析 2 个十六进制字符
        unsigned char hi = hex_char_to_val(piece[3]);
        unsigned char lo = hex_char_to_val(piece[4]);
        unsigned char byte_val = (hi << 4) | lo;

        // 返回对应的 byte_piece
        return (char *)(t->byte_pieces + byte_val * 2); // 每个占 2 字节
    }

    return piece;
}

void safe_printf(char *piece)
{
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL)
    {
        return;
    }
    if (piece[0] == '\0')
    {
        return;
    }
    if (piece[1] == '\0')
    {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val)))
        {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str}; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

char str_buffer[MAX_TOKEN_LENGTH * 2 + 3] __attribute__((aligned(4 * NR_LANES), section(".l2")));
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens)
{
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL)
    {
        // fprintf(stderr, "cannot encode NULL text\n");
        // exit(EXIT_FAILURE);
        printf("cannot encode NULL text\n");
    }

    if (t->sorted_vocab == NULL)
    {
        // t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));

        for (int i = 0; i < t->vocab_size; i++)
        {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    // char *str_buffer = malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos)
        tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0')
    {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++)
    {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80)
        {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
        {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1)
        {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        }
        else
        {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++)
            {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1)
    {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++)
        {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score)
            {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
        {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
        {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos)
        tokens[(*n_tokens)++] = 2;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct
{
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct
{
    int vocab_size;
    ProbIndex *probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n)
{
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
    {
        if (probabilities[i] > max_p)
        {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin)
{
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        if (coin < cdf)
        {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b)
{
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin)
{
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++)
    {
        if (probabilities[i] >= cutoff)
        {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++)
    {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp)
        {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++)
    {
        cdf += probindex[i].prob;
        if (r < cdf)
        {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

static ProbIndex sampler_probindex[VOCAB_SIZE] __attribute__((aligned(4 * NR_LANES), section(".l2")));

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed)
{
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    // sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
    sampler->probindex = sampler_probindex;
}

unsigned int random_u32(unsigned long long *state)
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state)
{ // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits)
{
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f)
    {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    }
    else
    {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++)
        {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1)
        {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        }
        else
        {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop
static int prompt_tokens[3] __attribute__((aligned(4 * NR_LANES), section(".l2")));

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
    printf("\n┌──────────────────────────────────┐\n");
    printf("│  Begin Generation                │\n");
    printf("│──────────────────────────────────│\n");
    printf("│ prompt: \"%12s\"           │\n", prompt ? prompt : "(null)");
    printf("│ steps: %-16d          │\n", steps);
    printf("└──────────────────────────────────┘\n");
    char *empty_prompt = "";
    if (prompt == NULL)
    {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    // int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    DEBUG_PRINT(3, "prompt: %s, num_prompt_tokens: %d\n", prompt, num_prompt_tokens);
    if (num_prompt_tokens < 1)
    {
        DEBUG_PRINT(3, "something is wrong, expected at least 1 prompt token\n");
    }

    // start the main loop
    int next;                     // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;                  // position in the sequence
    // start_timer();
    while (pos < steps)
    {
        DEBUG_PRINT(3, "\nBegin forward iteration, pos: %d, token: %d\n", pos, token);
        // forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // advance the state state machine
        if (pos < num_prompt_tokens - 1)
        {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        }
        else
        {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1)
        {
            break;
        }

        // print the token as string, decode it with the Tokenizer object
        char *piece = decode(tokenizer, token, next);
        DEBUG_PRINT(3, "token: %s", piece);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        token = next;

        // init the timer here because the first iteration can be slower
        // if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1)
    {
        // long end = time_in_ms();
        stop_timer();
        // fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
        int64_t runtime = get_timer();
        printf("generated toks: %d spent cycles: %ld\n", (pos - 1), runtime);
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

static inline unsigned long read_cycle()
{
    unsigned long cycle;
    asm volatile("rdcycle %0" : "=r"(cycle));
    return cycle;
}

int main()
{

    // default parameters
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 2;                   // number of steps to run for
    char *prompt = "";               // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // poor man's C argparse so we can override the defaults above from the command line

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = (unsigned int)read_cycle();
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer);
    if (steps == 0 || steps > transformer.config.seq_len)
        steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
    return 0;
}
#endif
