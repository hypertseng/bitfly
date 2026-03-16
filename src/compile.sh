rsync -av ./apps/ ../ara/apps/
rsync -av ./hardware/rtl/extended/* ../ara/hardware/
rsync -av ./hardware/rtl/bmpu/* ../ara/hardware/src/bmpu/
rsync -av ./Bender.yml ../ara/
cd ../ara/hardware/

make compile -j8
make verilate -j8