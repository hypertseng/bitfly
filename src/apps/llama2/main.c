/* Inference for Llama-2 Transformer model in pure C, int8 quantized forward pass. */
#ifndef LLAMA2_QUICK_COMPARE
#define LLAMA2_QUICK_COMPARE 1
#endif

#if !LLAMA2_QUICK_COMPARE
#include "runtime.h"
#include "util.h"
#include "model.h"
#include "kernel/bmpmm.h"
#include "tokenizer_data.h"
#include <string.h>
#include <float.h>

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

// 260K
// #define MAX_TOKEN_LENGTH 7
// #define VOCAB_SIZE 512
// #define DIM 64
// #define HIDDEN_DIM 172
// #define SEQ_LEN 512
// #define N_HEADS 8
// #define N_LAYERS 1 // 5
// #define N_KV_HEADS 4
// #define KV_DIM DIM *N_KV_HEADS / N_HEADS
// #define N_PROMPT_TOKENS 16

// 15m
#define MAX_TOKEN_LENGTH 27
#define VOCAB_SIZE 32000
#define DIM 288
#define HIDDEN_DIM 768
#define SEQ_LEN 256
#define N_HEADS 6
#define N_LAYERS 1 // 6
#define N_KV_HEADS 6
#define KV_DIM DIM *N_KV_HEADS / N_HEADS
#define N_PROMPT_TOKENS 16

// 42m
// #define MAX_TOKEN_LENGTH 27
// #define VOCAB_SIZE 32000
// #define DIM 512
// #define HIDDEN_DIM 1376
// #define SEQ_LEN 1024
// #define N_HEADS 8
// #define N_LAYERS 1 // 8
// #define N_KV_HEADS 8
// #define KV_DIM DIM *N_KV_HEADS / N_HEADS
// #define N_PROMPT_TOKENS 16

// // 110m
// #define MAX_TOKEN_LENGTH 27
// #define VOCAB_SIZE 32000
// #define DIM 768
// #define HIDDEN_DIM 2048
// #define SEQ_LEN 1024
// #define N_HEADS 12
// #define N_LAYERS 1 // 12
// #define N_KV_HEADS 12
// #define KV_DIM DIM *N_KV_HEADS / N_HEADS
// #define N_PROMPT_TOKENS 16
// #define TILE_TOKENS 8

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
    float x[DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float xb[DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float xb2[DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float hb[HIDDEN_DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float hb2[HIDDEN_DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    int8_t xq_q[DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float xq_s[DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    int8_t hq_q[HIDDEN_DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float hq_s[HIDDEN_DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float q[DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float k[KV_DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float v[KV_DIM * N_PROMPT_TOKENS]__attribute__((aligned(4 * NR_LANES)));
    float att[N_HEADS * SEQ_LEN]__attribute__((aligned(4 * NR_LANES)));
    float logits[N_PROMPT_TOKENS * VOCAB_SIZE]__attribute__((aligned(4 * NR_LANES)));
    float key_cache[N_LAYERS * SEQ_LEN * KV_DIM]__attribute__((aligned(4 * NR_LANES)));
    float value_cache[N_LAYERS * SEQ_LEN * KV_DIM]__attribute__((aligned(4 * NR_LANES)));
} StaticRunState;
StaticRunState run_state_buf __attribute__((aligned(4 * NR_LANES), section(".l2")));

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

// Transformer llama2 __attribute__((aligned(4 * NR_LANES), section(".l2")));

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

int64_t time_rmsnorm = 0;
int64_t time_softmax = 0;
int64_t time_matmul = 0;

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

// void quantize_batch(QuantizedTensor *qx, float *x, int batch, int dim)
// {
//     const int groups_per_row = dim / GS;
//     const float Q_MAX = 127.0f;

//     for (int b = 0; b < batch; b++)
//     {
//         float *x_row = x + b * dim;
//         int8_t *q_row = qx->q + b * dim;
//         float  *s_row = qx->s + b * groups_per_row;

//         for (int g = 0; g < groups_per_row; g++)
//         {
//             float wmax = 0.0f;
//             for (int i = 0; i < GS; i++)
//             {
//                 float val = fabsf(x_row[g * GS + i]);
//                 if (val > wmax) wmax = val;
//             }

//             float scale = wmax / Q_MAX;
//             s_row[g] = scale;

//             for (int i = 0; i < GS; i++)
//             {
//                 float quant = x_row[g * GS + i] / scale;
//                 int8_t qval = (int8_t)roundf(quant);
//                 q_row[g * GS + i] = qval;
//             }
//         }
//     }
// }

// void quantize_batch(QuantizedTensor *qx, float *x, int batch, int dim)
// {
//     const float Q_MAX = 127.0f;
//     const int total = batch * dim;

//     float wmax = 0.0f;
//     for (int i = 0; i < total; i++) {
//         float val = fabsf(x[i]);
//         if (val > wmax) wmax = val;
//     }

//     // 避免除0
//     float scale = (wmax > 1e-8f) ? (wmax / Q_MAX) : 1e-8f;
//     qx->s[0] = scale;

//     float inv_scale = 1.0f / scale;
//     for (int i = 0; i < total; i++) {
//         float quant = x[i] * inv_scale;
//         int32_t q = (int32_t)(quant + (quant >= 0 ? 0.5f : -0.5f));
//         if (q > 127) q = 127;
//         else if (q < -128) q = -128;
//         qx->q[i] = (int8_t)q;
//     }
// }

void quantize_batch(QuantizedTensor *qx, float *x, int batch, int dim) {
    const int total = batch * dim;
    const float Q_MAX = 127.0f;
    float max_val = 0.0f;

    // --------------------------
    // Step 1: 计算全局最大值 max(abs(x[i]))
    // --------------------------
    for (int i = 0; i < total;) {
        size_t vl;
        uint32_t local_max_bits;

        __asm__ volatile (
            "vsetvli t0, %[_n], e32, m1, ta, ma\n\t"        // 设置 VL
            "vle32.v v0, (%[x_ptr])\n\t"                    // 加载 float 数据 (按 uint32_t 处理)
            "li t1, 0x7fffffff\n\t"                         // 准备掩码：清除符号位
            "vmv.v.x v1, t1\n\t"                            // v1 = 0x7fffffff
            "vand.vv v0, v0, v1\n\t"                        // v0 = fabs(x) = x & 0x7fffffff
            "vredmax.vs v2, v0, v0\n\t"                     // 归约最大值到 v2[0]
            "vmv.x.s %[lmax], v2\n\t"                       // 提取归约结果
            "csrr %[vl], vl\n\t"                            // 获取当前 VL
            : [lmax] "=r"(local_max_bits), [vl] "=r"(vl)
            : [x_ptr] "r"(x + i), [_n] "r"(total - i)
            : "v0", "v1", "v2", "t0", "t1"
        );

        float local_max = *(float *)&local_max_bits;       // reinterpret bits back to float
        if (local_max > max_val) max_val = local_max;
        i += vl;
    }

    // --------------------------
    // Step 2: 计算 scale 和倒数
    // --------------------------
    float scale = (max_val > 1e-8f) ? (max_val / Q_MAX) : 1e-8f;
    float inv_scale = 1.0f / scale;
    qx->s[0] = scale;

    // --------------------------
    // Step 3: 量化：x[i] → q[i]
    // --------------------------
    for (int i = 0; i < total;) {
        size_t vl;

        float inv = inv_scale;
        __asm__ volatile (
            "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
            "vle32.v v0, (%[x_ptr])\n\t"            // 加载 x[i]
            "flw fa0, %[inv]\n\t"                   // 加载 inv_scale 到浮点寄存器
            "vfmul.vf v0, v0, fa0\n\t"              // 缩放 x[i]
            "vfcvt.x.f.v v1, v0\n\t"                // 转 int32 向零舍入
            "vse8.v v1, (%[q_ptr])\n\t"
            : [vl] "=r"(vl)
            : [x_ptr] "r"(x + i),
            [q_ptr] "r"(qx->q + i),
            [inv] "m"(inv),
            [n] "r"(total - i)
            : "v0", "v1", "fa0"
        );

        i += vl;
    }
}

// #define ALIGN_PTR(p, align) \
//     (void *)(((uintptr_t)(p) + (align - 1)) & ~(uintptr_t)(align - 1))

// /* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, QuantizedTensor *res, int n, int size_each)
{
    GS = n;
    void *p = *ptr;
    for (int i = 0; i < n; i++)
    {
        /* map quantized int8 values*/
        // p = ALIGN_PTR(p, 4 * NR_LANES);
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

    // dequantize(w->q_tokens, w->token_embedding_table, VOCAB_SIZE * DIM);

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
    printf("Begin building Transformer model\n");

    read_checkpoint(&t->config, &t->weights, &t->data, &t->file_size);

    printf("model config:\n");
    printf("dim: %d\n", t->config.dim);
    printf("hidden_dim: %d\n", t->config.hidden_dim);
    printf("n_layers: %d\n", t->config.n_layers);
    printf("n_heads: %d\n", t->config.n_heads);
    printf("n_kv_heads: %d\n", t->config.n_kv_heads);
    printf("vocab_size: %d\n", t->config.vocab_size);
    printf("seq_len: %d\n", t->config.seq_len);

    // StaticRunState run_state_buffer;
    malloc_run_state(&t->state, &run_state_buf);
    printf("RunState mallocated\n");
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float *o, float *x, float *weight, int size)
{
    start_timer();
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
        float a = weight[j];
        float b = ss * x[j];
        float tmp = a * b;
        o[j] = tmp;
    }
    stop_timer();
    time_rmsnorm += get_timer();
}

// void rmsnorm(float *o, float *x, float *weight, int size)
// {
//     float ss = 0.0f;
//     int i = 0;

//     // Part 1: Sum of squares reduction
//     asm volatile(
//         "vmv.v.x v0, zero\n\t" // v0 = 0
//         :
//         :
//         : "v0");

//     for (; i < size;)
//     {
//         size_t vl;
//         asm volatile(
//             "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
//             "vle32.v v1, (%[px])\n\t"
//             "vfmul.vv v2, v1, v1\n\t" // v2 = x * x
//             "vfadd.vv v0, v0, v2\n\t" // v0 += v2
//             : [vl] "=&r"(vl)
//             : [n] "r"(size - i), [px] "r"(x + i)
//             : "v0", "v1", "v2");
//         i += vl;
//     }

//     // Horizontal reduction to scalar
//     float sumsq;
//     asm volatile(
//         "vfredsum.vs v3, v0, v0\n\t" // v3[0] = sum(v0)
//         "vfmv.f.s %[sumsq], v3\n\t"
//         : [sumsq] "=f"(sumsq)
//         :
//         : "v0", "v3");

//     // Normalize factor
//     ss = sumsq / size + 1e-5f;
//     ss = 1.0f / sqrtf(ss);

//     // Part 2: Normalize and scale output
//     i = 0;
//     for (; i < size;)
//     {
//         size_t vl;
//         asm volatile(
//             "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
//             "vle32.v v1, (%[px])\n\t"       // load x
//             "vle32.v v2, (%[pw])\n\t"       // load weight
//             "vfmul.vf v3, v1, %[scale]\n\t" // v3 = x * ss
//             "vfmul.vv v3, v3, v2\n\t"       // v3 = v3 * weight
//             "vse32.v v3, (%[po])\n\t"       // store output
//             : [vl] "=&r"(vl)
//             : [n] "r"(size - i), [px] "r"(x + i), [pw] "r"(weight + i), [po] "r"(o + i), [scale] "f"(ss)
//             : "v1", "v2", "v3");
//         i += vl;
//     }
// }

// void softmax(float *x, int size)
// {
//     // find max value (for numerical stability)
//     float max_val = x[0];
//     for (int i = 1; i < size; i++)
//     {
//         if (x[i] > max_val)
//         {
//             max_val = x[i];
//         }
//     }
//     // exp and sum
//     float sum = 0.0f;
//     for (int i = 0; i < size; i++)
//     {
//         x[i] = expf(x[i] - max_val);
//         sum += x[i];
//     }
//     // normalize
//     for (int i = 0; i < size; i++)
//     {
//         x[i] /= sum;
//     }
// }

void softmax(float *x, int size)
{
    start_timer();
    printf("softmax test\n");
    // ---- Step 1: Find max value using RVV ----
    float max_val = -FLT_MAX;
    int i = 0;

    asm volatile(
        "vmv.v.x v0, %[init]\n\t"
        :
        : [init] "r"(-FLT_MAX)
        : "v0");

    for (; i < size;)
    {
        size_t vl;
        asm volatile(
            "vsetvli %[vl], %[n], e32, m1, ta, ma"
            : [vl] "=r"(vl)
            : [n] "r"(size - i));

        asm volatile(
            "vle32.v v1, (%[x])\n\t"
            "vfmax.vv v2, v1, v0\n\t"
            "vfredmax.vs v3, v2, v0\n\t"
            :
            : [x] "r"(x + i)
            : "v0", "v1", "v2", "v3", "t0");

        asm volatile(
            "vfmv.f.s %[res], v3\n\t"
            : [res] "=f"(max_val)
            :
            : "v3");

        i += vl;
    }

    // ---- Step 2: x[i] = exp(x[i] - max), sum all ----
    float sum = 0.0f;
    i = 0;
    for (; i < size;)
    {
        size_t vl;
        asm volatile(
            "vsetvli %[vl], %[n], e32, m1, ta, ma"
            : [vl] "=r"(vl)
            : [n] "r"(size - i));

        float temp_buf[vl];
        for (size_t j = 0; j < vl; j++)
        {
            temp_buf[j] = expf(x[i + j] - max_val);
            x[i + j] = temp_buf[j]; // store back to x
            sum += temp_buf[j];
        }

        i += vl;
    }

    // ---- Step 3: Normalize (x[i] /= sum) using RVV ----
    i = 0;
    for (; i < size;)
    {
        size_t vl;
        asm volatile(
            "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
            : [vl] "=r"(vl)
            : [n] "r"(size - i)
            : "t0");

        asm volatile(
            "vle32.v v0, (%[x])\n\t"
            "vfmul.vf v1, v0, %[invsum]\n\t"
            "vse32.v v1, (%[x])\n\t"
            :
            : [x] "r"(x + i), [invsum] "f"(1.0f / sum)
            : "v0", "v1", "memory");

        i += vl;
    }
    stop_timer();
    time_softmax += get_timer();
    printf("timesoft:%ld",time_softmax);
}

float freq_table[DIM / 2];
float cos_table[SEQ_LEN][DIM / 2];
float sin_table[SEQ_LEN][DIM / 2];

void transpose(const int8_t *A, int8_t *B, int M, int N)
{
    size_t vl;
    __asm__ volatile("vsetvli %[vl], %[m], e8, m1, ta, ma"
                     : [vl] "=&r"(vl)
                     : [m] "r"(M)
                     : "memory");

    const int TILE = 4; // 每次处理4列

    for (int col = 0; col < N; col += TILE)
    {
        int tile = (N - col >= TILE) ? TILE : (N - col);
        for (int i = 0; i < M; i += vl)
        {
            for (int t = 0; t < tile; ++t)
            {
                const int8_t *src = A + (i * N) + (col + t);
                int8_t *dst = B + (col + t) * M + i;

                // 可以展开为 v0~v3
                __asm__ volatile("vlse8.v v0, (%[src]), %[stride]"
                                 :
                                 : [src] "r"(src),
                                   [stride] "r"((ptrdiff_t)(N))
                                 : "memory", "v0");
                __asm__ volatile("vse8.v v0, (%[dst])"
                                 :
                                 : [dst] "r"(dst)
                                 : "memory", "v0");
            }
        }
    }
}


/**
 * 将任意大小的 int8 张量按每行 8 个 int8 打包为 uint64，
 * 每个块内部反转顺序（a0 在高位），最终按列优先顺序输出为 int8_t 字节数组。
 *
 * 参数:
 * - array: 输入的 int8_t 数组指针 (M x K)
 * - M: 行数
 * - K: 列数
 * - out: 输出缓冲区，必须足够大以容纳所有打包后的数据
 */



void pack_activation(const int8_t *array, int M, int K, int8_t *out) {
    const int tm = (M + 15) / 16;
    const int tk = (K > 256) ? ((K + 255) / 256) : 1;

    for (int i = 0; i < tm; i++) {
        const int m_start = i * 16;
        const int rows = (M - m_start < 16) ? (M - m_start) : 16;
        
        for (int j = 0; j < tk; j++) {
            const int k_start = j * 256;
            const int k_end = (k_start + 256 <= K) ? k_start + 256 : K;
            const int chunk_count = (k_end - k_start + 7) / 8;
            const int offset = (i * tk + j) * 16 * 256;
            
            const int8_t *block_in = array + offset;
            int8_t *out_block = out + offset;

            size_t vl;  // 声明vl变量
            asm volatile (
                "li t0, 0\n"                 // 初始化列计数器 (c)
                "1:\n"                       // 外层循环开始 (按列)
                "li t1, 0\n"                 // 初始化行计数器 (r)
                "2:\n"                       // 内层循环开始 (按行)
                
                // 设置向量长度
                "vsetvli %0, %1, e8, m1, ta, ma\n" 
                
                // 计算输入地址: block_in + c * 128 + r * 8
                "slli t2, t0, 7\n"           // t2 = c * 128
                "slli t3, t1, 3\n"          // t3 = r * 8
                "add t4, %2, t2\n"          // t4 = block_in + c * 128
                "add t4, t4, t3\n"         // t4 += r * 8
                
                // 加载向量
                "vle8.v v0, (t4)\n"         // 从输入地址加载向量
                
                // 计算输出地址: out_block + r * chunk_count * 8 + c * 8
                "mul t5, t1, %4\n"         // t5 = r * chunk_count
                "slli t5, t5, 3\n"          // t5 *= 8
                "slli t6, t0, 3\n"          // t6 = c * 8
                "add t5, t5, t6\n"          // t5 += c * 8
                "add t5, %3, t5\n"          // t5 = out_block + offset
                
                // 存储向量
                "vse8.v v0, (t5)\n"         // 存储向量到输出地址
                
                // 更新行计数器
                "add t1, t1, %0\n"         // r += vl
                "blt t1, %1, 2b\n"         // 如果 r < rows，继续内层循环
                
                // 更新列计数器
                "addi t0, t0, 1\n"         // c += 1
                "blt t0, %4, 1b\n"          // 如果 c < chunk_count，继续外层循环
                
                : "=&r" (vl)                // 输出操作数，绑定到vl变量
                : "r" (rows),               // 输入操作数1: rows
                  "r" (block_in),           // 输入操作数2: block_in
                  "r" (out_block),          // 输入操作数3: out_block
                  "r" (chunk_count)         // 输入操作数4: chunk_count
                : "t0", "t1", "t2", "t3", "t4", "t5", "t6", "v0", "memory"
            );
        }
    }
}

// #define TM 16
// #define TK 480

// void pack_activation(const int8_t *array, int M, int K, int8_t *out) {
//     const int tm = (M + TM - 1) / TM;
//     const int tk = (K > TK) ? ((K + TK - 1) / TK) : 1;

//     for (int i = 0; i < tm; ++i) {
//         const int m_start = i * TM;
//         const int rows = (M - m_start < TM) ? (M - m_start) : TM;

//         for (int j = 0; j < tk; ++j) {
//             const int k_start = j * TK;
//             const int k_len = (K - k_start < TK) ? (K - k_start) : TK;
//             const int chunk_count = (k_len + 7) / 8;  // 每个chunk是64位（8个int8）

//             const int64_t *block_in = (const int64_t *)(array + m_start * K + k_start * rows);
//             int64_t *out_block      = (int64_t *)(out   + m_start * K + k_start * rows);

//             const int row_stride_in  = chunk_count;  // 每行有多少chunk
//             const int row_stride_out = rows;         // 每列间距

//             int row = 0;
//             for (; row + 7 < rows; row += 8) {
//                 const int64_t *row_ptr[8];
//                 int64_t *col_ptr[8];
//                 for (int r = 0; r < 8; ++r) {
//                     row_ptr[r] = block_in + (row + r) * row_stride_in;
//                     col_ptr[r] = out_block + (row + r);
//                 }
//                 __asm__ volatile (
//                     "vsetvli t0, %[cnt], e64, m1, ta, ma\n\t"
//                     "vle64.v v0, (%[in0])\n\t"
//                     "vle64.v v1, (%[in1])\n\t"
//                     "vle64.v v2, (%[in2])\n\t"
//                     "vle64.v v3, (%[in3])\n\t"
//                     "vle64.v v4, (%[in4])\n\t"
//                     "vle64.v v5, (%[in5])\n\t"
//                     "vle64.v v6, (%[in6])\n\t"
//                     "vle64.v v7, (%[in7])\n\t"

//                     "vsse64.v v0, (%[out0]), %[stride]\n\t"
//                     "vsse64.v v1, (%[out1]), %[stride]\n\t"
//                     "vsse64.v v2, (%[out2]), %[stride]\n\t"
//                     "vsse64.v v3, (%[out3]), %[stride]\n\t"
//                     "vsse64.v v4, (%[out4]), %[stride]\n\t"
//                     "vsse64.v v5, (%[out5]), %[stride]\n\t"
//                     "vsse64.v v6, (%[out6]), %[stride]\n\t"
//                     "vsse64.v v7, (%[out7]), %[stride]\n\t"
//                     :
//                     : [cnt] "r"(chunk_count),
//                       [in0] "r"(row_ptr[0]), [in1] "r"(row_ptr[1]),
//                       [in2] "r"(row_ptr[2]), [in3] "r"(row_ptr[3]),
//                       [in4] "r"(row_ptr[4]), [in5] "r"(row_ptr[5]),
//                       [in6] "r"(row_ptr[6]), [in7] "r"(row_ptr[7]),
//                       [out0] "r"(col_ptr[0]), [out1] "r"(col_ptr[1]),
//                       [out2] "r"(col_ptr[2]), [out3] "r"(col_ptr[3]),
//                       [out4] "r"(col_ptr[4]), [out5] "r"(col_ptr[5]),
//                       [out6] "r"(col_ptr[6]), [out7] "r"(col_ptr[7]),
//                       [stride] "r"(row_stride_out * 8)
//                     : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
//                 );
//             }

//             // 处理剩余行（< 8）
//             for (; row < rows; ++row) {
//                 const int64_t *row_ptr = block_in + row * row_stride_in;
//                 int64_t *col_ptr       = out_block + row;

//                 __asm__ volatile (
//                     "vsetvli t0, %[cnt], e64, m1, ta, ma\n\t"
//                     "vle64.v v0, (%[in])\n\t"
//                     "vsse64.v v0, (%[out]), %[stride]\n\t"
//                     :
//                     : [cnt] "r"(chunk_count),
//                       [in] "r"(row_ptr),
//                       [out] "r"(col_ptr),
//                       [stride] "r"(row_stride_out * 8)
//                     : "t0", "v0", "memory"
//                 );
//             }
//         }
//     }
// }



// #define BLOCK_H 16
// #define BLOCK_W 32
// #define SUBTILE_W 4
// #define SUBTILES_PER_ROW (BLOCK_W / SUBTILE_W)  // 8
// #define SUBTILES_PER_COL (BLOCK_H)              // 16
// void unpack_output(const int16_t *input, int16_t *output, int H, int W) {
//     const int blocks_per_row = (W + BLOCK_W - 1) / BLOCK_W;
//     const int blocks_per_col = (H + BLOCK_H - 1) / BLOCK_H;

//     for (int by = 0; by < blocks_per_col; ++by) {
//         for (int bx = 0; bx < blocks_per_row; ++bx) {
//             int64_t block_offset = (by * blocks_per_row + bx) * BLOCK_H * BLOCK_W / 4;
//             const int64_t *block_in = (const int64_t *)input + block_offset;
//             int64_t *block_out = (int64_t *)output + block_offset;
            
//             long stride = BLOCK_W * sizeof(int16_t);

//             // 每列为一组，共 8 列，每列 16 行
//             __asm__ volatile (
//                 "vsetvli x0, %[vl], e64, m1, ta, ma\n"
//                 "mv t1, %[stride]\n"

//                 "vle64.v v0, (%[in0])\n"
//                 "vle64.v v1, (%[in1])\n"
//                 "vle64.v v2, (%[in2])\n"
//                 "vle64.v v3, (%[in3])\n"
//                 "vle64.v v4, (%[in4])\n"
//                 "vle64.v v5, (%[in5])\n"
//                 "vle64.v v6, (%[in6])\n"
//                 "vle64.v v7, (%[in7])\n"

//                 "vsse64.v v0, (%[out0]), t1\n"
//                 "vsse64.v v1, (%[out1]), t1\n"
//                 "vsse64.v v2, (%[out2]), t1\n"
//                 "vsse64.v v3, (%[out3]), t1\n"
//                 "vsse64.v v4, (%[out4]), t1\n"
//                 "vsse64.v v5, (%[out5]), t1\n"
//                 "vsse64.v v6, (%[out6]), t1\n"
//                 "vsse64.v v7, (%[out7]), t1\n"

//                 :
//                 : [vl] "r"(16),
//                   [stride] "r"(stride),
//                   [in0] "r"(block_in + 0 * SUBTILES_PER_COL),
//                   [in1] "r"(block_in + 1 * SUBTILES_PER_COL),
//                   [in2] "r"(block_in + 2 * SUBTILES_PER_COL),
//                   [in3] "r"(block_in + 3 * SUBTILES_PER_COL),
//                   [in4] "r"(block_in + 4 * SUBTILES_PER_COL),
//                   [in5] "r"(block_in + 5 * SUBTILES_PER_COL),
//                   [in6] "r"(block_in + 6 * SUBTILES_PER_COL),
//                   [in7] "r"(block_in + 7 * SUBTILES_PER_COL),

//                   [out0] "r"(block_out + 0),
//                   [out1] "r"(block_out + 1),
//                   [out2] "r"(block_out + 2),
//                   [out3] "r"(block_out + 3),
//                   [out4] "r"(block_out + 4),
//                   [out5] "r"(block_out + 5),
//                   [out6] "r"(block_out + 6),
//                   [out7] "r"(block_out + 7)
//                 : "t1", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
//             );
//         }
//     }
// }

// #define BLOCK_H 16
// #define BLOCK_W 32
// #define SUBTILE_W 4
// #define SUBTILES_PER_ROW (BLOCK_W / SUBTILE_W)  // 8
// #define SUBTILES_PER_COL (BLOCK_H)              // 16
// #define SUBTILES_PER_BLOCK (SUBTILES_PER_ROW * SUBTILES_PER_COL)  // 128

// void unpack_output(
//     const int16_t *input,
//     int16_t *output,
//     int H, int W
// ) {
//     const int blocks_per_row = (W + BLOCK_W - 1) / BLOCK_W;
//     const int blocks_per_col = (H + BLOCK_H - 1) / BLOCK_H;

//     for (int by = 0; by < blocks_per_col; ++by) {
//         for (int bx = 0; bx < blocks_per_row; ++bx) {
//             const int64_t *block_in = (const int64_t *)(input + (by * blocks_per_row + bx) * BLOCK_H * BLOCK_W);
//             int64_t *out_block_base = (int64_t *)(output + by * BLOCK_H * W + bx * BLOCK_W);
//             // start_timer();
//             for (int c = 0; c < SUBTILES_PER_COL; c += 8) {
//                 // 一次处理8列 × 8行 → tile = 8x8
//                 __asm__ volatile (
//                     "vsetvli t0, %[vl], e64, m1, ta, ma\n"
//                     // 加载 8 列
//                     "vle64.v v0,  (%[in0])\n"
//                     "vle64.v v1,  (%[in1])\n"
//                     "vle64.v v2,  (%[in2])\n"
//                     "vle64.v v3,  (%[in3])\n"
//                     "vle64.v v4,  (%[in4])\n"
//                     "vle64.v v5,  (%[in5])\n"
//                     "vle64.v v6,  (%[in6])\n"
//                     "vle64.v v7,  (%[in7])\n"

//                     // 写回每一行（共8行，每行含1个元素来自 v0~v7）
//                     // out[r * 16 + c + i] = v[i][r]

//                     "vse64.v v0, (%[out0])\n"
//                     "vse64.v v1, (%[out1])\n"
//                     "vse64.v v2, (%[out2])\n"
//                     "vse64.v v3, (%[out3])\n"
//                     "vse64.v v4, (%[out4])\n"
//                     "vse64.v v5, (%[out5])\n"
//                     "vse64.v v6, (%[out6])\n"
//                     "vse64.v v7, (%[out7])\n"

//                     :
//                     : [vl] "r"(8),
//                       [in0] "r"(block_in + (c + 0) * SUBTILES_PER_ROW),
//                       [in1] "r"(block_in + (c + 1) * SUBTILES_PER_ROW),
//                       [in2] "r"(block_in + (c + 2) * SUBTILES_PER_ROW),
//                       [in3] "r"(block_in + (c + 3) * SUBTILES_PER_ROW),
//                       [in4] "r"(block_in + (c + 4) * SUBTILES_PER_ROW),
//                       [in5] "r"(block_in + (c + 5) * SUBTILES_PER_ROW),
//                       [in6] "r"(block_in + (c + 6) * SUBTILES_PER_ROW),
//                       [in7] "r"(block_in + (c + 7) * SUBTILES_PER_ROW),

//                       [out0] "r"(out_block_base + 0 * SUBTILES_PER_COL + c),
//                       [out1] "r"(out_block_base + 1 * SUBTILES_PER_COL + c),
//                       [out2] "r"(out_block_base + 2 * SUBTILES_PER_COL + c),
//                       [out3] "r"(out_block_base + 3 * SUBTILES_PER_COL + c),
//                       [out4] "r"(out_block_base + 4 * SUBTILES_PER_COL + c),
//                       [out5] "r"(out_block_base + 5 * SUBTILES_PER_COL + c),
//                       [out6] "r"(out_block_base + 6 * SUBTILES_PER_COL + c),
//                       [out7] "r"(out_block_base + 7 * SUBTILES_PER_COL + c)
//                     : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
//                 );
//             }
//             // stop_timer();
//             // int64_t runtime = get_timer();
//             // printf("unpack timer cycles: %ld\n", runtime);
//         }
//     }
// }


#define BLOCK_H 16
#define BLOCK_W 32
#define SUBTILE_W 4
#define SUBTILES_PER_ROW (BLOCK_W / SUBTILE_W)  // 8
#define SUBTILES_PER_COL (BLOCK_H)              // 16
#define SUBTILES_PER_BLOCK (SUBTILES_PER_ROW * SUBTILES_PER_COL)  // 128

void unpack_output(
    const int16_t *input,
    int16_t *output,
    int H, int W
) {
    const int blocks_per_row = (W + BLOCK_W - 1) / BLOCK_W;
    const int blocks_per_col = (H + BLOCK_H - 1) / BLOCK_H;

    for (int by = 0; by < blocks_per_col; ++by) {
        for (int bx = 0; bx < blocks_per_row; ++bx) {
            const int64_t *block_in = (const int64_t *)(input + (by * blocks_per_row + bx) * BLOCK_H * BLOCK_W);
            int64_t *out_block_base = (int64_t *)(output + by * BLOCK_H * W + bx * BLOCK_W);
            // start_timer();
            for (int c = 0; c < SUBTILES_PER_COL; c += 8) {
                // 一次处理8列 × 8行 → tile = 8x8
                __asm__ volatile (
                    "vsetvli t0, %[vl], e64, m1, ta, ma\n"
                    // 加载 8 列
                    "vle64.v v0,  (%[in0])\n"
                    "vle64.v v1,  (%[in1])\n"
                    "vle64.v v2,  (%[in2])\n"
                    "vle64.v v3,  (%[in3])\n"
                    "vle64.v v4,  (%[in4])\n"
                    "vle64.v v5,  (%[in5])\n"
                    "vle64.v v6,  (%[in6])\n"
                    "vle64.v v7,  (%[in7])\n"

                    // 可选：进行寄存器级的shuffle转置（跳过，直接按列写回）
                    // 写回每一行（共8行，每行含1个元素来自 v0~v7）
                    // out[r * 16 + c + i] = v[i][r]

                    "vse64.v v0, (%[out0])\n"
                    "vse64.v v1, (%[out1])\n"
                    "vse64.v v2, (%[out2])\n"
                    "vse64.v v3, (%[out3])\n"
                    "vse64.v v4, (%[out4])\n"
                    "vse64.v v5, (%[out5])\n"
                    "vse64.v v6, (%[out6])\n"
                    "vse64.v v7, (%[out7])\n"

                    :
                    : [vl] "r"(8),
                      [in0] "r"(block_in + (c + 0) * SUBTILES_PER_ROW),
                      [in1] "r"(block_in + (c + 1) * SUBTILES_PER_ROW),
                      [in2] "r"(block_in + (c + 2) * SUBTILES_PER_ROW),
                      [in3] "r"(block_in + (c + 3) * SUBTILES_PER_ROW),
                      [in4] "r"(block_in + (c + 4) * SUBTILES_PER_ROW),
                      [in5] "r"(block_in + (c + 5) * SUBTILES_PER_ROW),
                      [in6] "r"(block_in + (c + 6) * SUBTILES_PER_ROW),
                      [in7] "r"(block_in + (c + 7) * SUBTILES_PER_ROW),

                      [out0] "r"(out_block_base + 0 * SUBTILES_PER_COL + c),
                      [out1] "r"(out_block_base + 1 * SUBTILES_PER_COL + c),
                      [out2] "r"(out_block_base + 2 * SUBTILES_PER_COL + c),
                      [out3] "r"(out_block_base + 3 * SUBTILES_PER_COL + c),
                      [out4] "r"(out_block_base + 4 * SUBTILES_PER_COL + c),
                      [out5] "r"(out_block_base + 5 * SUBTILES_PER_COL + c),
                      [out6] "r"(out_block_base + 6 * SUBTILES_PER_COL + c),
                      [out7] "r"(out_block_base + 7 * SUBTILES_PER_COL + c)
                    : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
                );
            }
            // stop_timer();
            // int64_t runtime = get_timer();
            // printf("unpack timer cycles: %ld\n", runtime);
        }
    }
}

static int8_t transposed_buffer[VOCAB_SIZE * HIDDEN_DIM] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int8_t packed_activation[VOCAB_SIZE * DIM] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int16_t matmul_out_buffer[VOCAB_SIZE * N_PROMPT_TOKENS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int16_t unpacked_out[VOCAB_SIZE  * N_PROMPT_TOKENS] __attribute__((aligned(4 * NR_LANES), section(".l2")));
void matmul_batch(float *xout, QuantizedTensor *x, QuantizedTensor *w, int seq_len, int n, int d)
{
    // printf("test\n");
    // printf("xout ptr = %p\n", (void *)xout);
    int64_t start = get_cycle_count();
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    // printf("matmul shape: W(%d, %d) @ x(%d,) -> xout(%d,)\n", d, n, n, d);
    // int8_t *transposed_weight = transposed_buffer;
    // start_timer();
    // transpose(w->q, transposed_weight, n, d);
    // stop_timer();
    // int64_t runtime = get_timer();
    // printf("mm1 timer cycles: %ld\n", runtime);
    // print matmul shape
    // printf("gemm shape: (%d, %d) (%d, %d)\n", seq_len, n, n ,d);
    // start_timer();
    pack_activation(x->q, seq_len, n, packed_activation);
    // printf("packed_activation ptr = %p\n", (void *)packed_activation);
    // printf("w ptr = %p\n", (void *)w);
    // printf("w->q ptr = %p\n", (void *)w->q);
    // stop_timer();
    // int64_t runtime = get_timer();
    // printf("mm2 timer cycles: %ld\n", runtime);
    // start_timer();
    binary_mixed_matmul(matmul_out_buffer, packed_activation, w->q, seq_len, n, d);
    // stop_timer();
    // runtime = get_timer();
    // printf("mm3 timer cycles: %ld\n", runtime);
    // start_timer();
    unpack_output(matmul_out_buffer, unpacked_out, seq_len, d);
    // stop_timer();
    // runtime = get_timer();
    // printf("mm4 timer cycles: %ld\n", runtime);

    // start_timer();
    // vector_int8_matmul(unpacked_out, x->q, transposed_buffer, seq_len, n, d);
    // stop_timer();
    // int64_t runtime = get_timer();
    // printf("mm4 timer cycles: %ld\n", runtime);

    // start_timer();
    // float scale_val = x->s[0] + w->s[0];
    // register float fscale asm("fa0") = scale_val;

    // __asm__ volatile(
    //     "vfmv.v.f v24, %[scale]\n"
    //     :
    //     : [scale] "f"(fscale)
    //     : "v24"
    // );

    // for (int l = 0; l < seq_len; l++) {
    //     int i = 0;
    //     int16_t* in_row = unpacked_out + l * d;
    //     float* out_row = xout + l * d;

    //     while (i < d) {
    //         size_t vl;
    //         __asm__ volatile(
    //             "vsetvli    %[vl], %[remain], e16, m8, ta, ma\n"
    //             "vle16.v    v16, (%[in])           \n"
    //             "vfcvt.f.x.v v16, v16              \n"
    //             "vfmul.vv   v16, v16, v24          \n"
    //             "vse32.v    v16, (%[out])          \n"
    //             : [vl] "=&r"(vl)
    //             : [in] "r"(in_row + i),
    //             [out] "r"(out_row + i),
    //             [remain] "r"(d - i)
    //             : "v16", "v24", "memory"
    //         );
    //         i += vl;
    //     }
    // }
    // stop_timer();
    // runtime = get_timer();
    // printf("mm5 timer cycles: %ld\n", runtime);
    // printf("mixed gemm timer cycles: %ld\n", runtime);
    // printf("RVV gemm timer cycles: %ld\n", runtime);
    time_matmul += get_cycle_count() - start;
    // printf("time_matmul:%ld\n",time_matmul);
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    // printf matmul shape
    // printf("matmul shape: W(%d, %d) @ x(%d,) -> xout(%d,)\n", d, n, n, d);

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }
}

static const float vhalf[VLEN/32] = { [0 ... VLEN/32-1] = 0.5f };
static const float vscale[VLEN/32] = { [0 ... VLEN/32-1] = 0.02f };

float *forward_prefill(Transformer *transformer, int *prompt_tokens, int num_prompt_tokens)
{
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    // printf("w ptr = %p\n", (void *)w);
    RunState *s = &transformer->state;

    int dim = DIM;
    int kv_dim = (DIM * N_KV_HEADS) / N_HEADS;
    int kv_mul = N_HEADS / N_KV_HEADS;
    int hidden_dim = HIDDEN_DIM;
    int head_size = dim / N_HEADS;

    float *x = s->x; // shape: [num_prompt_tokens, dim]

    // ─────────────────────────────
    // 1. Token embedding lookup
    for (int i = 0; i < num_prompt_tokens; i++) {
        int token = prompt_tokens[i];
        memcpy(x + i * dim, w->token_embedding_table + token * dim, dim * sizeof(float));
    }

    for (int i = 0; i < DIM / 2; i++) {
        int head_dim = i % (head_size / 2);
        freq_table[i] = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    }
    for (int t = 0; t < num_prompt_tokens; t++) {
        for (int i = 0; i < dim / 2; i++) {
            float val = t * freq_table[i];
            cos_table[t][i] = cosf(val);
            sin_table[t][i] = sinf(val);
        }
    }

    // ─────────────────────────────
    // 2. Layer-wise forward
    for (int l = 0; l < N_LAYERS; l++) {
        // printf("begin layer%d prefill forward\n", l);
        int64_t start = get_cycle_count();
        // ── 2.1 RMSNorm
        for (int i = 0; i < num_prompt_tokens; i++) {
            rmsnorm(s->xb + i * dim, x + i * dim, w->rms_att_weight + l * dim, dim);
        }
        int64_t runtime = get_cycle_count() - start;
        // printf("layer%d RMSNorm timer cycles: %ld\n", l, runtime);
        start = get_cycle_count();
        // ── 2.2 Q/K/V projection
        quantize_batch(&s->xq, s->xb, num_prompt_tokens, dim); // [T, dim] → quantized
        // printf("w->wq ptr = %p\n", (void *)w->wq);
        matmul_batch(s->q, &s->xq, w->wq + l, num_prompt_tokens, dim, dim);     // [T, dim]
        matmul_batch(s->k, &s->xq, w->wk + l, num_prompt_tokens, dim, kv_dim);  // [T, kv_dim]
        matmul_batch(s->v, &s->xq, w->wv + l, num_prompt_tokens, dim, kv_dim);  // [T, kv_dim]
        runtime = get_cycle_count() - start;
        printf("layer%d prefill forward QKV timer cycles: %ld\n", l, runtime);
        start = get_cycle_count();
        // ── 2.3 Apply RoPE positional encoding to Q & K
        // for (int t = 0; t < num_prompt_tokens; t++) {
        //     float *q_t = s->q + t * dim;
        //     float *k_t = s->k + t * kv_dim;

        //     for (int i = 0; i < dim; i += 2) {
        //         int head_dim = i % head_size;
        //         float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        //         float val = t * freq;
        //         float c = cosf(val), s_val = sinf(val);

        //         if (i + 1 >= dim) break;
        //         // rotate Q
        //         float v0 = q_t[i], v1 = q_t[i + 1];
        //         q_t[i]     = v0 * c - v1 * s_val;
        //         q_t[i + 1] = v0 * s_val + v1 * c;

        //         // rotate K (if within kv_dim)
        //         if (i < kv_dim) {
        //             float k0 = k_t[i], k1 = k_t[i + 1];
        //             k_t[i]     = k0 * c - k1 * s_val;
        //             k_t[i + 1] = k0 * s_val + k1 * c;
        //         }
        //     }
        // }
        // ── 2.3 Apply RoPE positional encoding using RISC-V Vector Extension (RVV) inline assembly
        // for (int t = 0; t < num_prompt_tokens; t++) {
        //     float *q_t = s->q + t * dim;
        //     float *k_t = s->k + t * kv_dim;
        //     float *cos_t = cos_table[t];
        //     float *sin_t = sin_table[t];

        //     for (int i = 0; i < dim; i += 2) {
        //         float c = cos_t[i / 2];
        //         float s_val = sin_t[i / 2];
        //         float v0 = q_t[i], v1 = q_t[i + 1];
        //         q_t[i]     = v0 * c - v1 * s_val;
        //         q_t[i + 1] = v0 * s_val + v1 * c;
        //         if (i < kv_dim) {
        //             float k0 = k_t[i], k1 = k_t[i + 1];
        //             k_t[i]     = k0 * c - k1 * s_val;
        //             k_t[i + 1] = k0 * s_val + k1 * c;
        //         }
        //     }
        // }
        for (int t = 0; t < num_prompt_tokens; t++) {
            float *q_t = s->q + t * dim;
            float *k_t = s->k + t * kv_dim;
            float *cos_t = cos_table[t];
            float *sin_t = sin_table[t];

            int i = 0;
            while (i < dim) {
                size_t vl;
                asm volatile(
                    "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
                    : [vl] "=r"(vl)
                    : [n] "r"((dim - i) / 2)
                );

                // Load q[i] and q[i+1]
                asm volatile(
                    "vle32.v v0, (%[in0])\n\t"   // v0 = q[i]
                    "vle32.v v1, (%[in1])\n\t"   // v1 = q[i+1]
                    :
                    : [in0] "r"(q_t + i), [in1] "r"(q_t + i + 1)
                    : "v0", "v1"
                );

                // Load cos and sin (they are 1 value per (i/2))
                asm volatile(
                    "vle32.v v2, (%[c_in])\n\t"  // v2 = cos_t[i/2]
                    "vle32.v v3, (%[s_in])\n\t"  // v3 = sin_t[i/2]
                    :
                    : [c_in] "r"(cos_t + i / 2), [s_in] "r"(sin_t + i / 2)
                    : "v2", "v3"
                );

                // v4 = v0 * v2 - v1 * v3
                asm volatile(
                    "vfmul.vv v4, v0, v2\n\t"
                    "vfmul.vv v5, v1, v3\n\t"
                    "vfsub.vv v4, v4, v5\n\t"
                );

                // v5 = v0 * v3 + v1 * v2
                asm volatile(
                    "vfmul.vv v6, v0, v3\n\t"
                    "vfmul.vv v7, v1, v2\n\t"
                    "vfadd.vv v5, v6, v7\n\t"
                );

                // Store rotated q back
                asm volatile(
                    "vse32.v v4, (%[out0])\n\t"
                    "vse32.v v5, (%[out1])\n\t"
                    :
                    : [out0] "r"(q_t + i), [out1] "r"(q_t + i + 1)
                    : "memory"
                );

                i += vl * 2;
            }

            // ---- Process k_t if necessary ----
            if (kv_dim > 0) {
                int j = 0;
                while (j < kv_dim) {
                    size_t vl_k;
                    asm volatile(
                        "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
                        : [vl] "=r"(vl_k)
                        : [n] "r"((kv_dim - j) / 2)
                    );

                    // Load k[j] and k[j+1]
                    asm volatile(
                        "vle32.v v0, (%[in0])\n\t"
                        "vle32.v v1, (%[in1])\n\t"
                        :
                        : [in0] "r"(k_t + j), [in1] "r"(k_t + j + 1)
                        : "v0", "v1"
                    );

                    // Load corresponding cos and sin
                    asm volatile(
                        "vle32.v v2, (%[c_in])\n\t"
                        "vle32.v v3, (%[s_in])\n\t"
                        :
                        : [c_in] "r"(cos_t + j / 2), [s_in] "r"(sin_t + j / 2)
                        : "v2", "v3"
                    );

                    // v4 = v0 * v2 - v1 * v3
                    asm volatile(
                        "vfmul.vv v4, v0, v2\n\t"
                        "vfmul.vv v5, v1, v3\n\t"
                        "vfsub.vv v4, v4, v5\n\t"
                    );

                    // v5 = v0 * v3 + v1 * v2
                    asm volatile(
                        "vfmul.vv v6, v0, v3\n\t"
                        "vfmul.vv v7, v1, v2\n\t"
                        "vfadd.vv v5, v6, v7\n\t"
                    );

                    // Store rotated k back
                    asm volatile(
                        "vse32.v v4, (%[out0])\n\t"
                        "vse32.v v5, (%[out1])\n\t"
                        :
                        : [out0] "r"(k_t + j), [out1] "r"(k_t + j + 1)
                        : "memory"
                    );

                    j += vl_k * 2;
                }
            }
        }
        runtime = get_cycle_count() - start;
        printf("layer%d prefill forward RoPE timer cycles: %ld\n", l, runtime);
        start = get_cycle_count();
        // ── 2.4 Save K/V to KV cache
        float *kcache = s->key_cache + l * p->seq_len * kv_dim;
        float *vcache = s->value_cache + l * p->seq_len * kv_dim;
        memcpy(kcache, s->k, num_prompt_tokens * kv_dim * sizeof(float));
        memcpy(vcache, s->v, num_prompt_tokens * kv_dim * sizeof(float));
        runtime = get_cycle_count() - start;
        printf("layer%d prefill forward KV cache timer cycles: %ld\n", l, runtime);
        start = get_cycle_count();
        // ── 2.5 Multi-head attention
        float scale = 1.0f / sqrtf(head_size);

        for (int h = 0; h < N_HEADS; h++)
        {
            float *q_h = s->q + h * head_size;
            float *k_h = s->key_cache + l * p->seq_len * kv_dim + (h / kv_mul) * head_size;
            float *att_h = s->att + h * p->seq_len * p->seq_len;

            // (1) Attention score：Q × K^T
            matmul_batch(att_h, (QuantizedTensor *)q_h, (QuantizedTensor *)k_h, num_prompt_tokens, p->seq_len, head_size);


            // (2) Scaled softmax with mask
            int64_t tmp_start = get_cycle_count();
            for (int t = 0; t < num_prompt_tokens; t++)
            {
                float *att_row = att_h + t * p->seq_len;

                for (int k = 0; k < p->seq_len; k++)
                {
                    if (k > t)
                    {
                        att_row[k] = -1e9f; // Causal Mask
                    }
                    else
                    {
                        att_row[k] *= scale;
                    }
                }

                float max_val = att_row[0];
                for (int k = 1; k <= t; k++)
                {
                    if (att_row[k] > max_val)
                        max_val = att_row[k];
                }

                float sum = 0.0f;
                for (int k = 0; k <= t; k++)
                {
                    att_row[k] = expf(att_row[k] - max_val);
                    sum += att_row[k];
                }

                for (int k = 0; k <= t; k++)
                {
                    att_row[k] /= sum;
                }
            }
            time_softmax += get_cycle_count() - tmp_start;

            // (3) Attention × Value → s->xb
            float *v_h = s->value_cache + l * p->seq_len * kv_dim + (h / kv_mul) * head_size;
            float *xb_h = s->xb + h * head_size;

            matmul_batch(xb_h, (QuantizedTensor *)att_h, (QuantizedTensor *)v_h, num_prompt_tokens, p->seq_len, head_size);
        }
        // ── 2.6 Project MHA output
        quantize_batch(&s->xq, s->xb, num_prompt_tokens, dim);
        matmul_batch(s->xb2, &s->xq, w->wo + l, num_prompt_tokens, dim, dim);
        for (int i = 0; i < num_prompt_tokens * dim; i++) {
            x[i] += s->xb2[i];
        }
        runtime = get_cycle_count() - start;
        printf("layer%d prefill forward attention timer cycles: %ld\n", l, runtime);
        start = get_cycle_count();
        // ── 2.7 FFN path
        for (int i = 0; i < num_prompt_tokens; i++) {
            rmsnorm(s->xb + i * dim, x + i * dim, w->rms_ffn_weight + l * dim, dim);
        }

        quantize_batch(&s->xq, s->xb, num_prompt_tokens, dim);
        matmul_batch(s->hb, &s->xq, w->w1 + l, num_prompt_tokens, dim, hidden_dim);
        matmul_batch(s->hb2, &s->xq, w->w3 + l, num_prompt_tokens, dim, hidden_dim);

        // for (int i = 0; i < num_prompt_tokens * hidden_dim; i++)
        // {
        //     float val = s->hb[i];
        //     val *= 1.0f / (1.0f + expf(-val)); // SiLU
        //     val *= s->hb2[i];                  // SwiGLU
        //     s->hb[i] = val;
        // }
        int i = 0;
        while (i < num_prompt_tokens * hidden_dim) {
            size_t vl;
            asm volatile (
                "vsetvli %[vl], %[n], e32, m1, ta, ma\n\t"
                "vle32.v v6, (%[vhalf])\n\t"
                "vle32.v v7, (%[vscale])\n\t"

                "vle32.v v0, (%[in])\n\t"

                "vfmul.vv v1, v0, v6\n\t"
                "vfmul.vv v2, v1, v1\n\t"
                "vfmul.vv v3, v2, v1\n\t"
                "vfmul.vv v4, v3, v7\n\t"
                "vfsub.vv v5, v1, v4\n\t"

                "vle32.v v6, (%[in2])\n\t"
                "vfmul.vv v0, v5, v6\n\t"
                "vse32.v v0, (%[out])\n\t"

                : [vl] "=r"(vl)
                : [n] "r"(num_prompt_tokens * hidden_dim - i),
                [vhalf] "r"(vhalf),
                [vscale] "r"(vscale),
                [in] "r"(s->hb + i),
                [in2] "r"(s->hb2 + i),
                [out] "r"(s->hb + i)
                : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
            );

            i += vl;
        }
        quantize_batch(&s->hq, s->hb, num_prompt_tokens, hidden_dim);;
        matmul_batch(s->xb, &s->hq, w->w2 + l, num_prompt_tokens, hidden_dim, dim);
        // for (int i = 0; i < num_prompt_tokens * dim; i++) {
        //     x[i] += s->xb[i];
        // }

        int total = num_prompt_tokens * dim;
        // start_timer();
        for (int i = 0; i < total; ) {
            size_t vl;
            asm volatile (
                "vsetvli %[vl], %[n], e32, m1, ta, ma"
                : [vl] "=r"(vl)
                : [n] "r"(total - i)
            );

            float *x_ptr  = x + i;
            float *xb_ptr = s->xb + i;

            // x[i:i+vl] += xb[i:i+vl]
            asm volatile (
                "vsetvli t0, %[vl], e32, m1, ta, ma\n\t"
                "vle32.v v0, (%[x])\n\t"       // v0 ← x
                "vle32.v v1, (%[xb])\n\t"      // v1 ← xb
                "vfadd.vv v2, v0, v1\n\t"      // v2 ← v0 + v1
                "vse32.v v2, (%[x])\n\t"       // x ← v2
                :
                : [x] "r"(x_ptr), [xb] "r"(xb_ptr), [vl] "r"(vl)
                : "v0", "v1", "v2", "t0", "memory"
            );

            i += vl;
        }
        // stop_timer();
        // runtime = get_timer();
        // printf("ffn test2: %ld\n", runtime);

        runtime = get_cycle_count() - runtime;
        printf("layer%d prefill forward FFN timer cycles: %ld\n", l, runtime);
    }

    // ─────────────────────────────
    // 3. Final RMSNorm and logits projection
    for (int i = 0; i < num_prompt_tokens; i++) {
        rmsnorm(x + i * dim, x + i * dim, w->rms_final_weight, dim);
    }
    quantize_batch(&s->xq, x, num_prompt_tokens, dim);
    matmul_batch(s->logits, &s->xq, w->wcls, num_prompt_tokens, dim, VOCAB_SIZE);
    return s->logits + (num_prompt_tokens - 1) * VOCAB_SIZE; // return final token logits
}

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
        printf("begin layer%d decode forward\n", l);
        // start_timer();
        // int64_t runtime = get_timer();
        // printf("layer%d cycle reset: %ld\n", l, runtime);
        DEBUG_PRINT(2, "├─ %d layer computation begin\n", l + 1);

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
        // qkv matmuls
        // runtime = get_timer();
        // printf("layer%d 1timer cycles: %ld\n", l, runtime);
        quantize(&s->xq, s->xb, dim);
        matmul_batch(s->q, &s->xq, w->wq + l, 1, dim, dim);
        matmul_batch(s->k, &s->xq, w->wk + l, 1, dim, kv_dim);
        matmul_batch(s->v, &s->xq, w->wv + l, 1, dim, kv_dim);
        DEBUG_PRINT(3, "│  ✓ Q/K/V computation finished\n");
        // runtime = get_timer();
        // printf("layer%d 2timer cycles: %ld\n", l, runtime);
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
            for (int i = 0; i < head_size; i++) xb[i] = 0.0f;
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
        // runtime = get_timer();
        // printf("layer%d 5timer cycles: %ld\n", l, runtime);
        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul_batch(s->xb2, &s->xq, w->wo + l, 1, dim, dim);
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
        matmul_batch(s->hb, &s->xq, w->w1 + l, 1, dim, hidden_dim);
        matmul_batch(s->hb2, &s->xq, w->w3 + l, 1, dim, hidden_dim);
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
        matmul_batch(s->xb, &s->hq, w->w2 + l, 1, hidden_dim, dim);
        DEBUG_PRINT(3, "│  ✓ FFN output matmul finished\n");
        // residual connection
        for (int i = 0; i < dim; i++)
        {
            x[i] += s->xb[i];
        }
        DEBUG_PRINT(2, "├─ %d layer computation finished\n", l + 1);
        // printf("layer%d cycles: %ld\n", l, get_cycle_count());
        // runtime = get_timer();
        // printf("layer%d timer cycles: %ld\n", l, runtime);
        // stop_timer();
        // runtime = get_timer();
        // printf("layer%d spent cycles: %ld\n", l, runtime);
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul_batch(s->logits, &s->xq, w->wcls, 1, dim, VOCAB_SIZE);
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
    char** vocab;
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

static TokenIndex sorted_vocab[VOCAB_SIZE];
static char* vocab_ptrs[VOCAB_SIZE];
void build_tokenizer(Tokenizer* t) {
    t->vocab_size = VOCAB_SIZE;

    // 遍历 vocab_data 构造 vocab[i]
    for (int i = 0; i < VOCAB_SIZE; i++) {
        int offset = vocab_offsets[i];
        int len = vocab_lengths[i];

        // vocab_ptrs[i] 指向 vocab_data 中 token 的起始位置
        vocab_ptrs[i] = (char*)&vocab_data[offset];

        // 确保 null-terminated（裸机运行时安全）
        if (vocab_data[offset + len] != '\0') {
            // vocab_data 在生成时就已经添加了 '\0'，这里是保险措施
            // 可以跳过或触发断言
        }
    }

    // 设置结构体字段
    t->vocab = vocab_ptrs;
    t->vocab_scores = (float*)vocab_scores;
    t->max_token_length = max_token_length;
    t->sorted_vocab = NULL;

    // 初始化 byte fallback
    for (int i = 0; i < 256; i++) {
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
        t->sorted_vocab = sorted_vocab;

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

    // printf("text: \"%s\"\n", text);
    // printf("prompt_length: %ld\n", strlen(text));
    // printf("bos: %d, eos: %d\n", bos, eos);
    // printf("tokens: ");
    // for (int i = 0; i < *n_tokens; i++) {
    //     printf("%d(%s) ", tokens[i], t->vocab[tokens[i]]);
    // }
    // printf("\nnum_tokens: %d\n", *n_tokens);
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
static int prompt_tokens[512] __attribute__((aligned(4 * NR_LANES), section(".l2")));

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps)
{
    printf("Begin Generation\n");
    printf("prompt: \"%s\"\n", prompt ? prompt : "(null)");
    printf("steps: %d\n", steps);
    char *empty_prompt = "";
    if (prompt == NULL)
    {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    // int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    // encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    // fix num_prompt_tokens for test
    num_prompt_tokens = 16;
    DEBUG_PRINT(3, "prompt: %s, num_prompt_tokens: %d\n", prompt, num_prompt_tokens);
    if (num_prompt_tokens < 1)
    {
        DEBUG_PRINT(3, "something is wrong, expected at least 1 prompt token\n");
    }
    printf("num_prompt_tokens: %d\n", num_prompt_tokens);
    start_timer();
    int64_t start_time = get_cycle_count();
    // prefill：一次性 forward 整个 prompt
    forward_prefill(transformer, prompt_tokens, num_prompt_tokens);
    printf("prefill finished\n");
    // 设置初始 token 和 pos
    int token = prompt_tokens[num_prompt_tokens - 1];
    int pos = num_prompt_tokens;
    // printf("pos:%d\n", pos);

    // decode 阶段：逐步 forward + sample
    while (pos < steps)
    {
        float *logits = forward(transformer, token, pos);
        int next = sample(sampler, logits);
        if (next == 1)
            break; // 终止条件

        char *piece = decode(tokenizer, token, next);
        printf("token: %d, piece: %s\n", next, piece);
        safe_printf(piece);

        token = next;
        pos++;
    }
    // // start the main loop
    // int next;                     // will store the next token in the sequence
    // int token = prompt_tokens[0]; // kick off with the first token in the prompt
    // int pos = 0;                  // position in the sequence
    // while (pos < steps)
    // {
    //     DEBUG_PRINT(3, "\nBegin forward iteration, pos: %d, token: %d\n", pos, token);
    //     // forward the transformer to get logits for the next token
    //     float *logits = forward(transformer, token, pos);
    //     // advance the state state machine
    //     if (pos < num_prompt_tokens - 1)
    //     {
    //         // if we are still processing the input prompt, force the next prompt token
    //         next = prompt_tokens[pos + 1];
    //     }
    //     else
    //     {
    //         // otherwise sample the next token from the logits
    //         next = sample(sampler, logits);
    //     }
    //     pos++;
    //     // data-dependent terminating condition: the BOS (=1) token delimits sequences
    //     if (next == 1)
    //     {
    //         break;
    //     }

    //     // print the token as string, decode it with the Tokenizer object
    //     char *piece = decode(tokenizer, token, next);
    //     DEBUG_PRINT(3, "token: %s", piece);
    //     printf("token: %d, piece: %s\n", next, piece);
    //     safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    //     token = next;
    // }
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1)
    {
        // long end = time_in_ms();
        int64_t stop_time = get_cycle_count();
        // fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
        int64_t runtime = stop_time - start_time;
        printf("\nGeneration finished\n");
        printf("generated toks: %d spent cycles: %ld\n", (pos - 1), runtime);
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

int main()
{

    // default parameters
    float temperature = 1.0f;        // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;               // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = N_PROMPT_TOKENS;                   // number of steps to run for
    char *prompt = "Once upon a time, there was a mountain, and";               // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // poor man's C argparse so we can override the defaults above from the command line

    // parameter validation/overrides
    if (rng_seed <= 0)
        rng_seed = get_cycle_count();
    if (temperature < 0.0)
        temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp)
        topp = 0.9;
    if (steps < 0)
        steps = 0;

    // build the Transformer via the model .bin file
    Transformer llama2;
    build_transformer(&llama2);
    if (steps == 0 || steps > llama2.config.seq_len)
        steps = llama2.config.seq_len; // override to ~max length
    // printf("ptr = %p\n", (void *)llama2.weights.wq);
    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, llama2.config.vocab_size, temperature, topp, rng_seed);

    // run!
    generate(&llama2, &tokenizer, &sampler, prompt, steps);
    printf("time_rmsnorm = %ld\n", time_rmsnorm);
    printf("time_softmax = %ld\n", time_softmax);
    printf("time_matmul = %ld\n", time_matmul);
    return 0;
}
#endif
#else

#include "runtime.h"
#include "util.h"
#include "kernel/bmpmm_compare.h"
#include "kernel/rvv_compare.h"
#include <stdint.h>
#include <string.h>

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define LLAMA2_COMPARE_M 128UL
#define LLAMA2_MAX_DIM 3072UL
#define LLAMA2_MAX_HIDDEN 8192UL
#define LLAMA2_MAX_N 8192UL
#define LLAMA2_MAX_K 8192UL
#define LLAMA2_MAX_ACT_BYTES (LLAMA2_COMPARE_M * LLAMA2_MAX_K)
#define LLAMA2_MAX_WEIGHT_BYTES (((LLAMA2_MAX_K + 7UL) / 8UL) * (((LLAMA2_MAX_N + 7UL) / 8UL) * 8UL))
#define LLAMA2_MAX_OUTPUT_ELEMS (LLAMA2_COMPARE_M * (((LLAMA2_MAX_N + 127UL) / 128UL) * 128UL))

static int8_t g_activation_buf[LLAMA2_MAX_ACT_BYTES] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int8_t g_weight_buf[LLAMA2_MAX_WEIGHT_BYTES] __attribute__((aligned(4 * NR_LANES), section(".l2")));
static int16_t g_output_buf[LLAMA2_MAX_OUTPUT_ELEMS] __attribute__((aligned(4 * NR_LANES), section(".l2")));

typedef struct
{
    const char *name;
    const char *asset_path;
    unsigned long dim;
    unsigned long hidden_dim;
    unsigned long n_layers;
    unsigned long n_heads;
    unsigned long n_kv_heads;
} LlamaQuickProfile;

typedef struct
{
    unsigned long M;
    unsigned long N;
    unsigned long K;
    llama_bmpmm_exec_cfg_t cfg;
} ShapeCfgEntry;

typedef struct
{
    const char *name;
    unsigned long M;
    unsigned long N;
    unsigned long K;
    llama_bmpmm_exec_cfg_t bmpmm_cfg;
    llama_rvv_exec_cfg_t rvv_cfg;
} BenchOp;

static const LlamaQuickProfile kProfiles[] = {
    {
        .name = "15M",
        .asset_path = "src/apps/llama2/15mmodel.bin",
        .dim = 288UL,
        .hidden_dim = 768UL,
        .n_layers = 6UL,
        .n_heads = 6UL,
        .n_kv_heads = 6UL,
    },
    {
        .name = "42M",
        .asset_path = "src/apps/llama2/stories42M.bin",
        .dim = 512UL,
        .hidden_dim = 1376UL,
        .n_layers = 8UL,
        .n_heads = 8UL,
        .n_kv_heads = 8UL,
    },
    {
        .name = "110M",
        .asset_path = "src/apps/llama2/stories110M.bin",
        .dim = 768UL,
        .hidden_dim = 2048UL,
        .n_layers = 12UL,
        .n_heads = 12UL,
        .n_kv_heads = 12UL,
    },
    {
        .name = "Llama-3.2-1B-q8_0",
        .asset_path = "src/apps/llama2/llama3.2.c/Llama-3.2-1B-q8_0.bin",
        .dim = 2048UL,
        .hidden_dim = 8192UL,
        .n_layers = 16UL,
        .n_heads = 32UL,
        .n_kv_heads = 8UL,
    },
    {
        .name = "Llama-3.2-3B-q8_0",
        .asset_path = "src/apps/llama2/llama3.2.c/Llama-3.2-3B-q8_0.bin",
        .dim = 3072UL,
        .hidden_dim = 8192UL,
        .n_layers = 28UL,
        .n_heads = 24UL,
        .n_kv_heads = 8UL,
    },
};

static const ShapeCfgEntry kShapeCfgs[] = {
    {.M = 128UL, .N = 288UL, .K = 288UL, .cfg = {16UL, 16UL, 64UL, 1UL, 4UL, 0UL}},
    {.M = 128UL, .N = 768UL, .K = 288UL, .cfg = {8UL, 16UL, 128UL, 1UL, 8UL, 0UL}},
    {.M = 128UL, .N = 288UL, .K = 768UL, .cfg = {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {.M = 128UL, .N = 512UL, .K = 512UL, .cfg = {8UL, 16UL, 128UL, 1UL, 8UL, 0UL}},
    {.M = 128UL, .N = 1376UL, .K = 512UL, .cfg = {8UL, 16UL, 128UL, 1UL, 8UL, 0UL}},
    {.M = 128UL, .N = 512UL, .K = 1376UL, .cfg = {8UL, 16UL, 128UL, 1UL, 8UL, 0UL}},
    {.M = 128UL, .N = 768UL, .K = 768UL, .cfg = {8UL, 16UL, 128UL, 1UL, 8UL, 0UL}},
    {.M = 128UL, .N = 2048UL, .K = 768UL, .cfg = {8UL, 16UL, 128UL, 1UL, 8UL, 0UL}},
    {.M = 128UL, .N = 768UL, .K = 2048UL, .cfg = {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {.M = 128UL, .N = 2048UL, .K = 2048UL, .cfg = {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {.M = 128UL, .N = 512UL, .K = 2048UL, .cfg = {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {.M = 128UL, .N = 8192UL, .K = 2048UL, .cfg = {16UL, 32UL, 64UL, 1UL, 2UL, 0UL}},
    {.M = 128UL, .N = 2048UL, .K = 8192UL, .cfg = {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {.M = 128UL, .N = 3072UL, .K = 3072UL, .cfg = {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {.M = 128UL, .N = 1024UL, .K = 3072UL, .cfg = {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {.M = 128UL, .N = 8192UL, .K = 3072UL, .cfg = {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
    {.M = 128UL, .N = 3072UL, .K = 8192UL, .cfg = {8UL, 128UL, 64UL, 1UL, 1UL, 0UL}},
};

static void init_buffers(void)
{
    static int initialized = 0;
    if (initialized)
        return;
    for (unsigned long i = 0; i < LLAMA2_MAX_ACT_BYTES; ++i)
        g_activation_buf[i] = (int8_t)((i & 0x7UL) - 3);
    for (unsigned long i = 0; i < LLAMA2_MAX_WEIGHT_BYTES; ++i)
        g_weight_buf[i] = (int8_t)(0x5a ^ (int)(i & 0x3fUL));
    for (unsigned long i = 0; i < LLAMA2_MAX_OUTPUT_ELEMS; ++i)
        g_output_buf[i] = 0;
    initialized = 1;
}

static const ShapeCfgEntry *find_shape_cfg(unsigned long M, unsigned long N, unsigned long K)
{
    for (unsigned long i = 0; i < (sizeof(kShapeCfgs) / sizeof(kShapeCfgs[0])); ++i)
    {
        const ShapeCfgEntry *entry = &kShapeCfgs[i];
        if (entry->M == M && entry->N == N && entry->K == K)
            return entry;
    }
    return 0;
}

static llama_rvv_exec_cfg_t default_rvv_cfg(void)
{
    llama_rvv_exec_cfg_t cfg = {
        .mtile = 8UL,
        .ntile = 64UL,
        .ktile = 128UL,
        .gm = 1UL,
        .gn = 2UL,
        .prec = 0UL,
    };
    return cfg;
}

static int build_profile_ops(const LlamaQuickProfile *profile, BenchOp *ops, int cap)
{
    const unsigned long kv_dim = (profile->dim * profile->n_kv_heads) / profile->n_heads;
    const struct
    {
        const char *name;
        unsigned long N;
        unsigned long K;
    } templates[] = {
        {"q_proj", profile->dim, profile->dim},
        {"k_proj", kv_dim, profile->dim},
        {"v_proj", kv_dim, profile->dim},
        {"o_proj", profile->dim, profile->dim},
        {"gate_proj", profile->hidden_dim, profile->dim},
        {"up_proj", profile->hidden_dim, profile->dim},
        {"down_proj", profile->dim, profile->hidden_dim},
    };

    if (cap < (int)(sizeof(templates) / sizeof(templates[0])))
        return 0;

    for (unsigned long i = 0; i < (sizeof(templates) / sizeof(templates[0])); ++i)
    {
        const ShapeCfgEntry *shape_cfg = find_shape_cfg(LLAMA2_COMPARE_M, templates[i].N, templates[i].K);
        if (!shape_cfg)
            return 0;
        ops[i].name = templates[i].name;
        ops[i].M = LLAMA2_COMPARE_M;
        ops[i].N = templates[i].N;
        ops[i].K = templates[i].K;
        ops[i].bmpmm_cfg = shape_cfg->cfg;
        ops[i].rvv_cfg = default_rvv_cfg();
    }
    return (int)(sizeof(templates) / sizeof(templates[0]));
}

static int64_t run_bmpmm_fast(const BenchOp *op)
{
    int64_t estimated_total = 0;
    llama_bmpmm_exec_opts_t opts = {
        .mode = BMPMM_LOWP_EXEC_FAST,
        .estimated_total_cycles = &estimated_total,
    };
    if (!llama_bmpmm_binary_with_cfg_opts(g_output_buf, g_activation_buf, g_weight_buf, op->M, op->K, op->N, &op->bmpmm_cfg, &opts))
        return -1;
    return estimated_total;
}

static int64_t run_rvv_fast(const BenchOp *op)
{
    int64_t estimated_total = 0;
    llama_rvv_exec_opts_t opts = {
        .mode = LLAMA2_RVV_EXEC_FAST,
        .estimated_total_cycles = &estimated_total,
    };
    if (!llama_rvv_binary_with_cfg_opts(g_output_buf, g_activation_buf, g_weight_buf, op->M, op->K, op->N, &op->rvv_cfg, &opts))
        return -1;
    return estimated_total;
}

int main(void)
{
    BenchOp ops[7];
    init_buffers();

    printf("[llama2] quick compare mode\n");
    printf("[llama2] offline BMPMM tiling search source: tmp/llama_quickcompare_best_binary.csv\n");
    printf("[llama2] proxy workload: first-layer GEMMs only, M=%lu tokens, excludes rope/softmax/norm/lm_head\n", LLAMA2_COMPARE_M);
    printf("[llama2] RVV baseline: current generic rvv_binary fast path (mtile=8, ntile=64, ktile=128, gm=1, gn=2)\n");

    for (unsigned long p = 0; p < (sizeof(kProfiles) / sizeof(kProfiles[0])); ++p)
    {
        const LlamaQuickProfile *profile = &kProfiles[p];
        const int op_count = build_profile_ops(profile, ops, 7);
        int64_t bmpmm_layer_cycles = 0;
        int64_t rvv_layer_cycles = 0;

        if (op_count <= 0)
        {
            printf("[llama2] ERROR: missing shape config for profile %s\n", profile->name);
            return 1;
        }

        printf("\n============================================================\n");
        printf("[llama2] model=%s\n", profile->name);
        printf("[llama2] source=%s\n", profile->asset_path);
        printf("[llama2] dim=%lu hidden=%lu layers=%lu heads=%lu kv_heads=%lu\n",
               profile->dim, profile->hidden_dim, profile->n_layers, profile->n_heads, profile->n_kv_heads);
        printf("============================================================\n");

        for (int i = 0; i < op_count; ++i)
        {
            const BenchOp *op = &ops[i];
            const int64_t bmpmm_cycles = run_bmpmm_fast(op);
            const int64_t rvv_cycles = run_rvv_fast(op);
            const double speedup = (bmpmm_cycles > 0 && rvv_cycles > 0) ? ((double)rvv_cycles / (double)bmpmm_cycles) : 0.0;

            if (bmpmm_cycles < 0 || rvv_cycles < 0)
            {
                printf("[llama2] ERROR: fast compare failed for %s shape=(%lu,%lu,%lu)\n",
                       op->name, op->M, op->N, op->K);
                return 1;
            }

            bmpmm_layer_cycles += bmpmm_cycles;
            rvv_layer_cycles += rvv_cycles;

            printf("[llama2] op=%-10s shape=(%lu,%lu,%lu) "
                   "bmpmm_cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu) "
                   "bmpmm_cycles=%ld rvv_cfg=(mt=%lu,nt=%lu,kt=%lu,gm=%lu,gn=%lu) "
                   "rvv_cycles=%ld speedup=%.3fx\n",
                   op->name,
                   op->M, op->N, op->K,
                   op->bmpmm_cfg.mtile, op->bmpmm_cfg.ntile, op->bmpmm_cfg.ktile, op->bmpmm_cfg.gm, op->bmpmm_cfg.gn,
                   (long)bmpmm_cycles,
                   op->rvv_cfg.mtile, op->rvv_cfg.ntile, op->rvv_cfg.ktile, op->rvv_cfg.gm, op->rvv_cfg.gn,
                   (long)rvv_cycles,
                   speedup);
        }

        printf("[llama2] layer_proxy bmpmm_cycles=%ld rvv_cycles=%ld speedup=%.3fx\n",
               (long)bmpmm_layer_cycles,
               (long)rvv_layer_cycles,
               rvv_layer_cycles > 0 ? (double)rvv_layer_cycles / (double)bmpmm_layer_cycles : 0.0);
        printf("[llama2] model_proxy bmpmm_cycles=%ld rvv_cycles=%ld speedup=%.3fx\n",
               (long)(bmpmm_layer_cycles * (int64_t)profile->n_layers),
               (long)(rvv_layer_cycles * (int64_t)profile->n_layers),
               rvv_layer_cycles > 0 ? (double)rvv_layer_cycles / (double)bmpmm_layer_cycles : 0.0);
    }

    return 0;
}

#endif
