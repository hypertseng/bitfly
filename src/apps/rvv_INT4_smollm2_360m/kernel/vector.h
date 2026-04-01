#ifndef BMPMM_INT4_VECTOR_H
#define BMPMM_INT4_VECTOR_H

#include <stdint.h>

#define RVV_INT4_VECTOR_EXEC_STRICT 0UL
#define RVV_INT4_VECTOR_EXEC_FAST 1UL

#ifndef RVV_INT4_VECTOR_DEFAULT_MODE
#define RVV_INT4_VECTOR_DEFAULT_MODE RVV_INT4_VECTOR_EXEC_STRICT
#endif

typedef struct
{
    // `fast` estimates runtime from sampled RVV block execution.
    // It does not materialize a valid full output matrix.
    unsigned long mode;
    int64_t *estimated_total_cycles;
} rvv_int4_vector_exec_opts_t;

void vector_int4_matmul(int16_t *restrict c, const int8_t *restrict a,
                        const int8_t *restrict b,
                        unsigned long int M, unsigned long int K,
                        unsigned long int N);
void vector_int4_matmul_with_opts(int16_t *restrict c, const int8_t *restrict a,
                                  const int8_t *restrict b,
                                  unsigned long int M, unsigned long int K,
                                  unsigned long int N,
                                  const rvv_int4_vector_exec_opts_t *opts);
void vector_int4_set_default_mode(unsigned long mode);
unsigned long vector_int4_get_default_mode(void);
int64_t vector_int4_get_last_estimated_total_cycles(void);
int64_t vector_int4_get_last_estimated_compute_cycles(void);

extern int64_t vector_compute_time;

#endif // BMPMM_INT4_VECTOR_H
