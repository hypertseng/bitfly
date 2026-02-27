# QLoMA

## Compilation
```bash
git submodule update --init --recursive --depth 1
git submodule sync --recursive
# 1. 复制 RISCVAsmParser.cpp
cp -f ./src/instr/RISCVAsmParser.cpp /data2/zzx/data/workspace/QLoMA/ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/AsmParser/RISCVAsmParser.cpp

# 2. 复制 RISCVDisassembler.cpp
cp -f ./src/instr/RISCVDisassembler.cpp /data2/zzx/data/workspace/QLoMA/ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/Disassembler/RISCVDisassembler.cpp

# 3. 复制 RISCVInstPrinter.cpp
cp -f ./src/instr/RISCVInstPrinter.cpp /data2/zzx/data/workspace/QLoMA/ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/RISCVInstPrinter.cpp

# 4. 复制 RISCVInstPrinter.h
cp -f ./src/instr/RISCVInstPrinter.h /data2/zzx/data/workspace/QLoMA/ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/MCTargetDesc/RISCVInstPrinter.h

# 5. 复制 RISCVInstrInfoCustom.td
cp -f ./src/instr/RISCVInstrInfoCustom.td /data2/zzx/data/workspace/QLoMA/ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/RISCVInstrInfoCustom.td

# 6. 复制 RISCVInstrInfo.td
cp -f ./src/instr/RISCVInstrInfo.td /data2/zzx/data/workspace/QLoMA/ara/toolchain/riscv-llvm/llvm/lib/Target/RISCV/RISCVInstrInfo.td

cd src/
sh compile.sh
```


## Run
```bash
cd QLoMA/ara/apps
make bin/customapp
cd ../hardware
make app=customapp sim
```

## questasim
```bash
sudo apt install libxft2 libxft2:i386 lib32ncurses6
sudo apt install libxext6
sudo apt install libxext6:i386

python2 mgclicgen.py <mac_address>

find $QUESTA_HOME -name salt_mgls_asynch -exec sh -c './pubkey_verify -y $0' {} \;

export PATH="/data2/zzx/data/eda/questasim/linux_x86_64":$PATH
export PATH="/data2/zzx/data/eda/questasim/RUVM_2021.2":$PATH
export SALT_LICENSE_SERVER="/data2/zzx/data/eda/questasim/license.dat":$SALT_LICENSE_SERVER
```


## expand dram
L2NumWords in ara_soc.sv