import torch
import numpy as np

# 设置随机种子以确保结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def emit_int8_col_major(name, array, alignment="NR_LANES*4"):
    """将 int8 数组按列优先方式打包成 uint32 输出为 .word 十六进制格式"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    # 按列优先展平数组
    flat_col_major = array.t().contiguous().flatten().numpy().astype(np.int8)

    # 填充到 4 的倍数长度
    padded = np.pad(
        flat_col_major, (0, (4 - len(flat_col_major) % 4) % 4), constant_values=0
    )

    # 打包为 uint32
    packed = np.frombuffer(padded.tobytes(), dtype=np.uint32)

    for i in range(0, len(packed), 8):
        line = ", ".join([f"0x{v:08x}" for v in packed[i : i + 8]])
        print(f"    .word {line}")


def emit_int8_packed_activations(name, array):
    """
    将任意大小的 int8 张量按每行 8 个 int8 打包为一个 uint64，
    每个块内部反转顺序（a0 在高位），最终按列优先顺序输出为 .quad 十六进制格式
    """

    # 转换为 numpy 并确保是 int8 类型
    if isinstance(array, torch.Tensor):
        array = array.numpy().astype(np.int8)

    M, N = array.shape  # M 行 N 列

    print(f".global {name}")
    print(f".balign 8")
    print(f"{name}:")

    blocks = []

    for i in range(M):
        row = array[i]

        # 如果当前行不足 8 的倍数，填充到下一个 8 的倍数
        pad_len = (8 - len(row) % 8) % 8
        padded_row = np.pad(row, (0, pad_len), constant_values=0)

        # 拆分为多个 8 字节 chunk，并打包为 uint64（反转字节顺序）
        for chunk_start in range(0, len(padded_row), 8):
            chunk = padded_row[chunk_start : chunk_start + 8]
            reversed_chunk = chunk[::-1]  # 反转顺序：a7,a6,...,a0
            packed = np.frombuffer(reversed_chunk.tobytes(), dtype=np.uint64)[0]
            blocks.append(packed)

    # 构造虚拟二维块矩阵：M 行 × chunks_per_row 列
    chunks_per_row = len(padded_row) // 8
    block_matrix = np.reshape(blocks, (M, chunks_per_row))

    # 转置矩阵得到列优先顺序
    col_major_blocks = block_matrix.T.flatten()

    # 输出每个 .quad
    for i in range(0, len(col_major_blocks), 4):  # 每行放 4 个 quad 提高可读性
        line_vals = []
        for j in range(i, min(i + 4, len(col_major_blocks))):
            val = col_major_blocks[j]
            line_vals.append(f"0x{val:016x}")
        print(f"    .quad {', '.join(line_vals)}")


def emit_int16_col_major(name, array, alignment="NR_LANES*4"):
    """将 int16 数组按列优先方式打包成 uint32 输出为 .word 十六进制格式"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")
    # 按列优先展平数组
    flat_col_major = array.t().contiguous().flatten().numpy().astype(np.int16)
    # 填充到 4 的倍数长度
    padded = np.pad(
        flat_col_major, (0, (4 - len(flat_col_major) % 4) % 4), constant_values=0
    )
    # 打包为 uint32
    packed = np.frombuffer(padded.tobytes(), dtype=np.uint32)
    for i in range(0, len(packed), 8):
        line = ", ".join([f"0x{v:08x}" for v in packed[i : i + 8]])
        print(f"    .word {line}")


def emit_int8_row_major(name, array, alignment="NR_LANES*4"):  # <<< 新增：按行优先方式
    """将 int8 数组按行优先方式打包成 uint32 输出为 .word 十六进制格式"""
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    # 按行优先展平数组
    flat_row_major = array.contiguous().flatten().numpy().astype(np.int8)

    # 填充到 4 的倍数长度
    padded = np.pad(
        flat_row_major, (0, (4 - len(flat_row_major) % 4) % 4), constant_values=0
    )

    # 打包为 uint32
    packed = np.frombuffer(padded.tobytes(), dtype=np.uint32)

    for i in range(0, len(packed), 8):
        line = ", ".join([f"0x{v:08x}" for v in packed[i : i + 8]])
        print(f"    .word {line}")


def emit_int32(name, array, alignment="NR_LANES*4"):
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    flat = array.contiguous().flatten().numpy().astype(np.int32)

    # 转为 uint32 表示避免符号扩展问题
    packed = np.frombuffer(flat.tobytes(), dtype=np.uint32)

    for i in range(0, len(packed), 8):
        line = ", ".join([f"0x{v:08x}" for v in packed[i : i + 8]])
        print(f"    .word {line}")


def emit_1bit_packed_weights(name, tensor, alignment="NR_LANES*4"):
    """
    将每个 8x8 的 1-bit block 打包为 64-bit，然后拆成两个 .word 输出（十六进制格式）
    """
    assert tensor.dim() == 2
    K, N = tensor.shape
    assert K % 8 == 0 and N % 8 == 0

    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    for i_block in range(0, K, 8):
        for j_block in range(0, N, 8):
            block = tensor[i_block : i_block + 8, j_block : j_block + 8]
            bits = 0
            for col in range(8):
                for row in range(8):
                    bit = int(block[row, col]) & 1
                    bits |= bit << (col * 8 + row)
            # 拆分为低32位和高32位
            low = bits & 0xFFFFFFFF
            high = (bits >> 32) & 0xFFFFFFFF

            # 输出为十六进制
            print(f"    .word 0x{low:08x}, 0x{high:08x}")


# 示例参数
M, N = 16, 32
K_dim = [16, 32, 64, 128, 256, 480]

for K in K_dim:
    # 创建矩阵
    # A = torch.ones((M, K), dtype=torch.int32)
    # B = torch.ones((K, N), dtype=torch.int32)
    A = torch.randint(-128, 127, (M, K), dtype=torch.int32)
    B = torch.randint(0, 2, (K, N), dtype=torch.int32)
    # A 每行值从 0 一直增加

    # data for debug
    # A = torch.arange(0, M * K, dtype=torch.int32).reshape(M, K)
    # B = torch.ones((K, N), dtype=torch.int32)
    # B = torch.zeros((K, N), dtype=torch.int32)
    # 发射数据段
    emit_int8_packed_activations(f"activation_lp_K_{K}", A)
    emit_1bit_packed_weights(f"weight_lp_K_{K}", B)
    emit_int16_col_major(f"result_lp_K_{K}", torch.zeros((M, N), dtype=torch.int16))

    emit_int8_row_major(f"activation_hp_K_{K}", A)
    B = torch.where(B == 0, -1, B)
    emit_int8_row_major(f"weight_hp_K_{K}", B)
    emit_int32(f"result_hp_K_{K}", torch.zeros((M, N), dtype=torch.int32))

    C = torch.zeros((M, N), dtype=torch.int32)
    C = A @ B
    # 打印A的第一列，按十六进制输出
    # print(f"A[0, :] = {[f'0x{v:02x}' for v in A[0, :]]}")
    # print(A[0, :])
    # print(B[:, 0])
    # print(C)
    # for m in range(M):
    #     for n in range(N):
    #         acc = 0
    #         for k in range(K):
    #             a_val = int(A[m, k])
    #             b_val = int(B[k, n])
    #             acc += a_val if b_val == 1 else -a_val
    #         C[m, n] = acc

    emit_int32(f"result_torch_K_{K}", C)
