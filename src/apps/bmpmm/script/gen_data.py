import torch
import math
import numpy as np

# 设置随机种子以确保结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def emit_int8_packed_activations(name, array):
    """
    将任意大小的 int8 张量按每行 8 个 int8 打包为一个 uint64，
    每个块内部反转顺序（a0 在高位），最终按列优先顺序输出为 .quad 十六进制格式
    """

    # 转换为 numpy 并确保是 int8 类型
    if isinstance(array, torch.Tensor):
        array = array.numpy().astype(np.int8)

    M, K = array.shape  # M 行 K 列

    print(f".global {name}")
    print(f".balign 8")
    print(f"{name}:")

    tm = math.ceil(M / 16.0)
    tk = 1
    if K > 480:
        tk = math.ceil(K / 480.0)
    for i in range(tm):
        m = min(16, M - i * 16)
        for j in range(tk):
            k = min(480, K - j * 480)
            blocks = []
            for r in range(16):
                if r + 1 > m:  # pad zero row
                    row = np.zeros(k, dtype=np.int8)
                else:
                    row = array[i * 16 + r][j * 480 : j * 480 + k]
                # 如果当前行不足 8 的倍数，填充到下一个 8 的倍数
                pad_len = (8 - len(row) % 8) % 8
                padded_row = np.pad(row, (0, pad_len), constant_values=0)

                # 拆分为多个 8 字节 chunk，并打包为 uint64
                for chunk_start in range(0, len(padded_row), 8):
                    chunk = padded_row[chunk_start : chunk_start + 8]
                    reversed_chunk = chunk[::-1]
                    packed = np.frombuffer(reversed_chunk.tobytes(), dtype=np.uint64)[0]
                    blocks.append(packed)

            # 构造二维块矩阵：M 行 × chunks_per_row 列
            chunks_per_row = len(padded_row) // 8
            block_matrix = np.reshape(blocks, (16, chunks_per_row))

            # 转置矩阵得到列优先顺序
            col_major_blocks = block_matrix.T.flatten()

            # 输出每个 .quad
            for block in range(0, len(col_major_blocks), 4):
                line_vals = []
                for j in range(block, min(block + 4, len(col_major_blocks))):
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


def emit_int16_row_major(name, array, alignment="NR_LANES*4"):
    """
    将 int16 数组按行优先方式打包成 uint32 输出为 .word 十六进制格式
    :param name: 符号名称
    :param array: torch.Tensor 或 numpy.ndarray，int16 类型矩阵
    :param alignment: 对齐方式（默认适用于 RVV NR_LANES*4 字节）
    """
    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    # 保证是 numpy 的 int16 类型
    flat_row_major = array.contiguous().flatten().numpy().astype(np.int16)

    # 补齐长度为4的倍数（4个 int16 = 2 个 uint32）
    padded = np.pad(
        flat_row_major, (0, (4 - len(flat_row_major) % 4) % 4), constant_values=0
    )

    # 每两个 int16 组成一个 uint32
    packed = np.frombuffer(padded.tobytes(), dtype=np.uint32)

    # 每行打印8个 uint32，即 8 * 4 字节 = 32 字节
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


def emit_1bit_packed_weights(name, array, alignment="NR_LANES*4"):
    """
    将每个 8x8 的 1-bit block 打包为 64-bit，然后拆成两个 .word 输出（十六进制格式）。
    如果块不足 8x8，则进行 zero-padding。
    """
    assert array.dim() == 2
    K, N = array.shape

    print(f".global {name}")
    print(f".balign {alignment}")
    print(f"{name}:")

    tn = math.ceil(N / 32.0)
    tk = math.ceil(K / 480.0)

    for i in range(tn):
        n = min(32, N - i * 32)
        for j in range(tk):
            k = min(480, K - j * 480)
            for i_block in range(0, k, 8):
                for j_block in range(0, n, 8):
                    bits = 0
                    for col in range(8):
                        for row in range(8):
                            global_row = j * 480 + i_block + row
                            global_col = i * 32 + j_block + col
                            if global_row < K and global_col < N:
                                bit = int(array[global_row, global_col]) & 1
                            else:
                                bit = 0  # padding with 0
                            bits |= bit << (col * 8 + row)
                    low = bits & 0xFFFFFFFF
                    high = (bits >> 32) & 0xFFFFFFFF
                    print(f"    .word 0x{low:08x}, 0x{high:08x}")


# 示例参数
M, N = 16, 32
K_dim = [8, 16, 32, 64, 128, 256, 480]
# M_dim = [1, 16, 32, 64, 128, 256, 512]
# M_dim = [50]
# N = 50
# K = 500

for K in K_dim:
    # 创建矩阵
    # A = torch.ones((M, K), dtype=torch.int16)
    # B = torch.ones((K, N), dtype=torch.int16)
    A = torch.randint(-128, 127, (M, K), dtype=torch.int16)
    B = torch.randint(0, 2, (K, N), dtype=torch.int16)
    # A 每行值从 0 一直增加

    # data for debug
    # A = torch.arange(0, M * K, dtype=torch.int16).reshape(M, K)  # A[0] must in int8 scope
    # B = torch.ones((K, N), dtype=torch.int16)
    # B = torch.zeros((K, N), dtype=torch.int16)
    # 发射数据段
    emit_int8_packed_activations(f"activation_lp_len_{K}", A)
    emit_1bit_packed_weights(f"weight_lp_len_{K}", B)
    emit_int16_col_major(f"result_lp_len_{K}", torch.zeros((M, N), dtype=torch.int16))

    emit_int8_row_major(f"activation_hp_len_{K}", A)
    B = torch.where(B == 0, -1, B)
    emit_int8_row_major(f"weight_hp_len_{K}", B)
    emit_int16_row_major(f"result_hp_len_{K}", torch.zeros((M, N), dtype=torch.int16))

    C = torch.zeros((M, N), dtype=torch.int16)
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
    # print(f"A = {A.tolist()}")
    # print(f"B = {B.tolist()}")
    # print(f"C = {C.tolist()}")
    # x = A[0, :K]
    # y = B[:K, 0]
    # # A 的第一行
    # print(f"x = {x.tolist()}")
    # # A 的第一行（十六进制显示）
    # print(f"x_hex = {[f'0x{(v & 0xFF):02x}' for v in x.tolist()]}")

    # # B 的第一列
    # print(f"y = {y.tolist()}")
    # # 计算x y同位置相乘再相加，每八个元素一组，一共8个
    # for i in range(0, K, 8):
    #     acc = 0
    #     for j in range(8):
    #         if i + j < K:
    #             a_val = int(x[i + j])
    #             b_val = int(y[i + j])
    #             acc += a_val if b_val == 1 else -a_val
    #     print(f"acc[{i // 8}] = {acc}")
    # z = x @ y

    emit_int16_row_major(f"result_torch_len_{K}", C)

