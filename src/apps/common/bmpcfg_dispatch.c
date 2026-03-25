#include <stddef.h>
#include <stdint.h>
#include "bmpcfg_dispatch.h"

typedef void (*bmpcfg_stub_fn_t)(void);

typedef struct
{
    uint32_t imm25;
    bmpcfg_stub_fn_t fn;
} bmpcfg_dispatch_entry_t;

#include "bmpcfg_dispatch_table.inc"

static int bmpcfg_try_pack(unsigned long prec, unsigned long K,
                           unsigned long mtile, unsigned long ntile,
                           unsigned long gm, unsigned long gn,
                           uint32_t *imm25)
{
    if (!imm25)
        return 0;

    if (prec > 7UL)
        return 0;

    if (K < 8UL || K > 256UL || (K & 7UL) != 0)
        return 0;

    if (gm < 1UL || gm > 8UL)
        return 0;

    if (gn < 1UL || gn > 8UL)
        return 0;

    if (mtile < 8UL || mtile > 64UL || ((mtile - 8UL) & 3UL) != 0)
        return 0;

    if (ntile < 16UL || ntile > 128UL || ((ntile - 16UL) & 15UL) != 0)
        return 0;

    *imm25 = ((uint32_t)prec << 22) |
             ((uint32_t)((K >> 3) - 1UL) << 17) |
             ((uint32_t)((mtile - 8UL) >> 2) << 13) |
             ((uint32_t)((ntile - 16UL) >> 4) << 10) |
             ((uint32_t)(gm - 1UL) << 7) |
             ((uint32_t)(gn - 1UL) << 4);
    return 1;
}

int bmpcfg_emit_prec(unsigned long prec, unsigned long K,
                     unsigned long mtile, unsigned long ntile,
                     unsigned long gm, unsigned long gn)
{
    uint32_t key;
    size_t lo = 0;
    size_t hi = sizeof(bmpcfg_dispatch_table) / sizeof(bmpcfg_dispatch_table[0]);

    if (!bmpcfg_try_pack(prec, K, mtile, ntile, gm, gn, &key))
        return 0;

    while (lo < hi)
    {
        size_t mid = lo + (hi - lo) / 2;
        if (bmpcfg_dispatch_table[mid].imm25 < key)
            lo = mid + 1;
        else
            hi = mid;
    }

    if (lo >= sizeof(bmpcfg_dispatch_table) / sizeof(bmpcfg_dispatch_table[0]) ||
        bmpcfg_dispatch_table[lo].imm25 != key)
        return 0;

    bmpcfg_dispatch_table[lo].fn();
    return 1;
}
