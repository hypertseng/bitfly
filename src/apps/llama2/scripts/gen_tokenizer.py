import struct

VOCAB_SIZE = 32000
INPUT_PATH = "/home/data/zzx/workspace/QLoMA/src/apps/llama2/tokenizer.bin"
OUTPUT_PATH = (
    "/home/data/zzx/workspace/QLoMA/src/apps/llama2/tokenizer_data.h"
)

with open(INPUT_PATH, "rb") as f:
    max_token_length = struct.unpack("i", f.read(4))[0]
    scores = []
    lengths = []
    offsets = []
    data = bytearray()
    offset = 0

    for _ in range(VOCAB_SIZE):
        score = struct.unpack("f", f.read(4))[0]
        scores.append(score)
        length = struct.unpack("i", f.read(4))[0]
        token = f.read(length)
        data += token + b"\x00"  # 添加 null terminator
        lengths.append(length)
        offsets.append(offset)
        offset += length + 1

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    out.write("#ifndef TOKENIZER_DATA_H\n#define TOKENIZER_DATA_H\n\n")
    out.write(f"#define VOCAB_SIZE {VOCAB_SIZE}\n")
    out.write(f"#define MAX_VOCAB_BYTES {len(data)}\n\n")

    out.write(f"const unsigned int max_token_length = {max_token_length};\n\n")

    out.write("const float vocab_scores[VOCAB_SIZE] = {\n")
    for s in scores:
        out.write(f"    {s:.8f}f,\n")
    out.write("};\n\n")

    out.write("const int vocab_lengths[VOCAB_SIZE] = {\n")
    for l in lengths:
        out.write(f"    {l},\n")
    out.write("};\n\n")

    out.write("const int vocab_offsets[VOCAB_SIZE] = {\n")
    for o in offsets:
        out.write(f"    {o},\n")
    out.write("};\n\n")

    out.write(f"const char vocab_data[MAX_VOCAB_BYTES] = {{\n")
    for b in data:
        c = chr(b) if 32 <= b <= 126 and chr(b) != "'" else "."
        out.write(f"    {b}, // '{c}'\n")
    out.write("};\n\n")

    out.write("#endif // TOKENIZER_DATA_H\n")
