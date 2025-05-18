# QLoMA

## Compilation
submodule update --init --recursive
git submodule sync --recursive
sh compile.sh

## Run
cd QLoMA/ara/apps
make bin/customapp
cd ../hardware
make app=customapp sim