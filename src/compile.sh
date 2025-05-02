rsync -av ./hardware/rtl/extended/* ../ara/hardware/
rsync -av ./hardware/rtl/mpu/* ../ara/hardware/src/mpu/
rsync -av ./Bender.yml ../ara/
rsync -av ./customapp/ ../ara/apps/customapp/

cd ../ara/hardware/

make compile