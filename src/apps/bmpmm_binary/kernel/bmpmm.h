#ifndef _BMPMM_H_
#define _BMPMM_H_

#include <stdint.h>
#include <string.h>
#include "runtime.h"

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// helpers for working with the vector unit
#define MAX_VLEN_BITS VLEN
#define MAX_VLEN_BYTES (MAX_VLEN_BITS / 8)
#define MAX_INT8_VL (MAX_VLEN_BITS / 8)
#define MAX_INT16_VL (MAX_VLEN_BITS / 16)

/*
 * public headers that expose the various kernels and the data helper
 * structure.  clients may simply include <kernel/bmpmm.h> and get access
 * to everything they need.
 */
#include "data.h"
#include "mixed.h"

#endif // BMPMM_H
