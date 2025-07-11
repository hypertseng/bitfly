def bin_to_header(bin_file, header_file):
    with open(bin_file, "rb") as f:
        data = f.read()

    with open(header_file, "w") as f:
        # Write header guard and includes
        f.write("#ifndef MODEL_H\n")
        f.write("#define MODEL_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write("#include <stddef.h>\n\n")

        # Write model data as byte array
        f.write("static const unsigned char model_bin[] = {\n")
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            f.write("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n")
        f.write("};\n\n")

        f.write(f"static const size_t model_bin_len = {len(data)};\n\n")
        f.write("#endif // MODEL_H\n")


if __name__ == "__main__":
    bin_to_header("../15mmodel.bin", "../model.h")
