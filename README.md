# QLoMA

## Compilation
```bash
submodule update --init --recursive
git submodule sync --recursive
sh compile.sh
```

## Run
```bash
cd QLoMA/ara/apps
make bin/customapp
cd ../hardware
make app=customapp sim
```