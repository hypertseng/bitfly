import torch
import numpy as np


# 设置随机种子以确保结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def emit_int8(name, array, alignment="4"):
    """将 int8 数组按 4 个一组打包成 uint32，输出为 .word 十六进制格式"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")
    flat = array.flatten().numpy().astype(np.int8)
    padded = np.pad(flat, (0, (4 - len(flat) % 4) % 4), constant_values=0)
    # 使用 uint32，避免溢出
    packed = np.frombuffer(padded.tobytes(), dtype=np.uint32)
    for i in range(0, len(packed), 8):
        line = ", ".join([f"0x{v:08x}" for v in packed[i : i + 8]])
        print(f"    .word {line}")


def emit_int32(name, array, alignment="4"):
    """直接按 .word 输出 int32 数组，使用 uint32 避免溢出"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")
    flat = array.flatten().numpy().astype(np.int32)
    # 将 int32 视为 uint32，防止溢出
    as_uint32 = np.frombuffer(flat.tobytes(), dtype=np.uint32)
    for i in range(0, len(as_uint32), 8):
        line = ", ".join([f"0x{v:08x}" for v in as_uint32[i : i + 8]])
        print(f"    .word {line}")


def emit_1bit_blocks(name, tensor, alignment="4"):
    """
    将每个 8x8 的 1-bit block 打包为 64-bit，然后拆成两个 .word 输出（十六进制）
    """
    assert tensor.dim() == 2
    K, N = tensor.shape
    assert K % 8 == 0 and N % 8 == 0

    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    for j_block in range(0, N, 8):
        for i_block in range(0, K, 8):
            block = tensor[i_block : i_block + 8, j_block : j_block + 8]
            bits = 0
            for col in range(8):
                for row in range(8):
                    bit = int(block[row, col]) & 1
                    bits |= bit << (col * 8 + row)
            low = bits & 0xFFFFFFFF
            high = (bits >> 32) & 0xFFFFFFFF
            print(f"    .word 0x{low:08x}, 0x{high:08x}")


# 示例参数
M, K, N = 16, 64, 32
A = torch.randint(-128, 127, (M, K), dtype=torch.int8)
B = torch.randint(0, 2, (K, N), dtype=torch.uint8)

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

print('.section .data,"aw",@progbits')

emit_int8("activation_int8", A)
emit_1bit_blocks("weight_int1", B)
emit_int8("weight_int8", B)
emit_int8("result_int8", result_int8)
emit_int32("result_int32", result_int32)
emit_int32("standard_int8", C)
