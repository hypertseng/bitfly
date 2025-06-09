import struct

with open("../tok512.bin", "rb") as f:
    max_token_length = struct.unpack("i", f.read(4))[0]
    vocab_size = 512  # 你需要根据实际设定

    scores = []
    strings = []
    lengths = []

    for _ in range(vocab_size):
        score = struct.unpack("f", f.read(4))[0]
        strlen = struct.unpack("i", f.read(4))[0]
        s_bytes = f.read(strlen)
        scores.append(score)
        strings.append(s_bytes)
        lengths.append(strlen)

with open("../tokenizer.h", "w", encoding="utf-8") as out:
    out.write(f"#pragma once\n\n")
    out.write(f"#define TOKENIZER_VOCAB_SIZE {vocab_size}\n")
    out.write(f"#define TOKENIZER_MAX_TOKEN_LENGTH {max_token_length}\n\n")

    # 写入 scores
    out.write(f"static const float tokenizer_vocab_scores[{vocab_size}] = {{\n")
    for score in scores:
        out.write(f"    {score:.8f}f,\n")
    out.write("};\n\n")

    # 写入 vocab（二进制字节数组）
    out.write(
        f"static const unsigned char tokenizer_vocab[{vocab_size}][TOKENIZER_MAX_TOKEN_LENGTH] = {{\n"
    )
    for s_bytes in strings:
        # 用 hex 形式输出，长度不足补 0
        padded = s_bytes.ljust(max_token_length, b"\0")
        hex_bytes = ", ".join(f"0x{b:02x}" for b in padded)
        out.write(f"    {{ {hex_bytes} }},\n")
    out.write("};\n\n")

    # 写入 lengths 数组
    out.write(f"static const size_t tokenizer_vocab_lens[{vocab_size}] = {{\n")
    for length in lengths:
        out.write(f"    {length},\n")
    out.write("};\n")
