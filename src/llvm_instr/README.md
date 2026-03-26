# LLVM Instruction Overlays

`src/llvm_instr/` contains the LLVM-side changes required for bitfly custom instruction support.

## Files

- `RISCVAsmParser.cpp`: assembler parsing changes
- `RISCVDisassembler.cpp`: disassembler decoding changes
- `RISCVInstPrinter.cpp` / `.h`: textual printing changes
- `RISCVInstrInfo.td`: instruction definitions
- `RISCVInstrInfoCustom.td`: custom instruction description fragments
- `RISCVMCCodeEmitter.cpp`: encoding emission changes
- `encoding.out.h`: generated or reference encoding header
- `rv_custom`: encoding reference data

## How It Is Used

This directory is not synced into Ara automatically unless you provide a destination:

```bash
scripts/dev/sync_src_to_ara.sh --llvm-dst <llvm-riscv-root>
```

Keep this directory aligned with the exact LLVM checkout you build against.
