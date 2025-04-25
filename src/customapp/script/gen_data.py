import torch
import numpy as np


def emit_int8(name, array, alignment="8"):
    """Emit int8 activation matrix as .byte"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")
    flat = array.flatten().numpy().astype(np.int8)
    for i in range(0, len(flat), 16):  # 每行输出最多16个byte
        line = ", ".join([f"{v}" for v in flat[i : i + 16]])
        print(f"    .byte {line}")

def emit_int32(name, array, alignment="8"):
    """Emit int32 activation matrix as .word"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")
    flat = array.flatten().numpy().astype(np.int32)
    for i in range(0, len(flat), 8):  # 每行输出最多8个word
        line = ", ".join([f"{v}" for v in flat[i : i + 8]])
        print(f"    .word {line}")


def emit_1bit_blocks(name, tensor, alignment="8"):
    """
    按 column-major 顺序打包 8x8 的权重 block。
    每个 block 生成一个 64-bit .quad
    """
    assert tensor.dim() == 2
    K, N = tensor.shape
    assert K % 8 == 0 and N % 8 == 0, "Weight matrix size must be divisible by 8"

    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    for j_block in range(0, N, 8):
        for i_block in range(0, K, 8):
            block = tensor[i_block : i_block + 8, j_block : j_block + 8]  # 取 8×8 小块
            bits = 0
            for col in range(8):
                for row in range(8):
                    bit = int(block[row, col]) & 1
                    bits |= bit << (
                        col * 8 + row
                    )  # 列优先展开：bit[pos] = block[row, col]
            print(f"    .quad 0x{bits:016x}")


# 示例参数
M, K, N = 16, 64, 32  # A: (M,K), B: (K,N)

# Activation: int8
A = torch.randint(-128, 127, (M, K), dtype=torch.int8)

# Weight: 1-bit (0 or 1)，用bool/uint8模拟
B = torch.randint(0, 2, (K, N), dtype=torch.uint8)

# 计算自定义乘法结果 C
C = torch.zeros((M, N), dtype=torch.int32)
for m in range(M):
    for n in range(N):
        acc = 0
        for k in range(K):
            a_val = int(A[m, k])
            b_val = int(B[k, n])
            acc += a_val if b_val == 1 else -a_val
        C[m, n] = acc

result_int8 = torch.zeros((M, N), dtype=torch.int8)

result_int32 = torch.zeros((M, N), dtype=torch.int32)

# 输出为 RISC-V 汇编格式
print('.section .data,"aw",@progbits')

# 输出 activation
emit_int8("activation_int8", A)

# 输出 weight（bit-packed）
emit_1bit_blocks("weight_int1", B)

emit_int8("weight_int8", B)

# 输出结果
emit_int8("result_int8", result_int8)

emit_int32("result_int32", result_int32)

# 正确答案
emit_int8("standard_int8", C)
