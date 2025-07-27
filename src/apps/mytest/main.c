#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define TM 16
#define TK 480

void pack_activation(const int8_t *array, int M, int K, int8_t *out) {
    const int tm = (M + TM - 1) / TM;
    const int tk = (K > TK) ? ((K + TK - 1) / TK) : 1;

    for (int i = 0; i < tm; ++i) {
        const int m_start = i * TM;
        const int rows = (M - m_start < TM) ? (M - m_start) : TM;

        for (int j = 0; j < tk; ++j) {
            const int k_start = j * TK;
            const int k_len = (K - k_start < TK) ? (K - k_start) : TK;
            const int chunk_count = (k_len + 7) / 8;  // 每个chunk是64位（8个int8）

            const int64_t *block_in = (const int64_t *)(array + m_start * K + k_start * rows);
            int64_t *out_block      = (int64_t *)(out   + m_start * K + k_start * rows);

            const int row_stride_in  = chunk_count;  // 每行有多少chunk
            const int row_stride_out = rows;         // 每列间距

            int row = 0;
            for (; row + 7 < rows; row += 8) {
                const int64_t *row_ptr[8];
                int64_t *col_ptr[8];
                for (int r = 0; r < 8; ++r) {
                    row_ptr[r] = block_in + (row + r) * row_stride_in;
                    col_ptr[r] = out_block + (row + r);
                }
                __asm__ volatile (
                    "vsetvli t0, %[cnt], e64, m1, ta, ma\n\t"
                    "vle64.v v0, (%[in0])\n\t"
                    "vle64.v v1, (%[in1])\n\t"
                    "vle64.v v2, (%[in2])\n\t"
                    "vle64.v v3, (%[in3])\n\t"
                    "vle64.v v4, (%[in4])\n\t"
                    "vle64.v v5, (%[in5])\n\t"
                    "vle64.v v6, (%[in6])\n\t"
                    "vle64.v v7, (%[in7])\n\t"

                    "vsse64.v v0, (%[out0]), %[stride]\n\t"
                    "vsse64.v v1, (%[out1]), %[stride]\n\t"
                    "vsse64.v v2, (%[out2]), %[stride]\n\t"
                    "vsse64.v v3, (%[out3]), %[stride]\n\t"
                    "vsse64.v v4, (%[out4]), %[stride]\n\t"
                    "vsse64.v v5, (%[out5]), %[stride]\n\t"
                    "vsse64.v v6, (%[out6]), %[stride]\n\t"
                    "vsse64.v v7, (%[out7]), %[stride]\n\t"
                    :
                    : [cnt] "r"(chunk_count),
                      [in0] "r"(row_ptr[0]), [in1] "r"(row_ptr[1]),
                      [in2] "r"(row_ptr[2]), [in3] "r"(row_ptr[3]),
                      [in4] "r"(row_ptr[4]), [in5] "r"(row_ptr[5]),
                      [in6] "r"(row_ptr[6]), [in7] "r"(row_ptr[7]),
                      [out0] "r"(col_ptr[0]), [out1] "r"(col_ptr[1]),
                      [out2] "r"(col_ptr[2]), [out3] "r"(col_ptr[3]),
                      [out4] "r"(col_ptr[4]), [out5] "r"(col_ptr[5]),
                      [out6] "r"(col_ptr[6]), [out7] "r"(col_ptr[7]),
                      [stride] "r"(row_stride_out * 8)
                    : "t0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "memory"
                );
            }

            // 处理剩余行（< 8）
            for (; row < rows; ++row) {
                const int64_t *row_ptr = block_in + row * row_stride_in;
                int64_t *col_ptr       = out_block + row;

                __asm__ volatile (
                    "vsetvli t0, %[cnt], e64, m1, ta, ma\n\t"
                    "vle64.v v0, (%[in])\n\t"
                    "vsse64.v v0, (%[out]), %[stride]\n\t"
                    :
                    : [cnt] "r"(chunk_count),
                      [in] "r"(row_ptr),
                      [out] "r"(col_ptr),
                      [stride] "r"(row_stride_out * 8)
                    : "t0", "v0", "memory"
                );
            }
        }
    }
}

