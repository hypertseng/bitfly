#ifndef BMPMM_DATA_DECLS_H
#define BMPMM_DATA_DECLS_H

/*
 * Central place to declare which pre-generated data blobs should be
 * referenced by the C sources.  By editing this file you control which
 * extern symbols are declared and which branches inside get_kernel_data()
 * are compiled into the binary.
 *
 * Usage: uncomment or add the `DECLARE_*` lines for the datasets you want
 * to compile in, and keep the rest commented out to save memory / binary
 * size. For each `DECLARE_KERNEL_DATA_SQUARE(<s>)` also define the
 * corresponding `HAS_KERNEL_DATA_SQUARE_<s>` so `data.c` will include the
 * matching `case`.
 */

/* Current validation scope: enable only one representative shape (64x64). */
// DECLARE_KERNEL_DATA_SQUARE(16)
// #define HAS_KERNEL_DATA_SQUARE_16 1
// DECLARE_KERNEL_DATA_SQUARE(32)
// #define HAS_KERNEL_DATA_SQUARE_32 1
DECLARE_KERNEL_DATA_SQUARE(64)
#define HAS_KERNEL_DATA_SQUARE_64 1
// DECLARE_KERNEL_DATA_SQUARE(128)
// #define HAS_KERNEL_DATA_SQUARE_128 1
// DECLARE_KERNEL_DATA_SQUARE(256)
// #define HAS_KERNEL_DATA_SQUARE_256 1
// DECLARE_KERNEL_DATA_SQUARE(512)
// #define HAS_KERNEL_DATA_SQUARE_512 1
// DECLARE_KERNEL_DATA_SQUARE(1024)
// #define HAS_KERNEL_DATA_SQUARE_1024 1
// DECLARE_KERNEL_DATA_SQUARE(2048)
// #define HAS_KERNEL_DATA_SQUARE_2048 1
// DECLARE_KERNEL_DATA_SQUARE(4096)
// #define HAS_KERNEL_DATA_SQUARE_4096 1


DECLARE_KERNEL_DATA(case1)
#define HAS_KERNEL_DATA_CASE1 1
DECLARE_KERNEL_DATA(case2)
#define HAS_KERNEL_DATA_CASE2 1
DECLARE_KERNEL_DATA(case3)
#define HAS_KERNEL_DATA_CASE3 1
DECLARE_KERNEL_DATA(case4)
#define HAS_KERNEL_DATA_CASE4 1
DECLARE_KERNEL_DATA(case5)
#define HAS_KERNEL_DATA_CASE5 1

#endif
