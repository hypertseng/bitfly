rsync -av ./customapp/ ../ara/apps/customapp/
rsync -av ./hardware/rtl/extended/* ../ara/hardware/
rsync -av ./hardware/rtl/mpu/* ../ara/hardware/src/mpu/
rsync -av ./Bender.yml ../ara/
cd ../ara/hardware/

make compile