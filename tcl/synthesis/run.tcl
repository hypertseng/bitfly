#-------------------Specify Libraries------------------
set target_library "/PATH/skywater-pdk/db/sky130_fd_sc_hd__tt_025C_1v80.db"
set link_library "* /PATH/skywater-pdk/db/sky130_fd_sc_hd__tt_025C_1v80.db"

source ../scripts/ara.tcl

set_host_options -max_cores 3

elaborate ara_soc -parameter "NrLanes = 4, VLEN = 4096"
uniquify
current_design ara_00000004_00001000_00000001_38_1_1_1_00000080_00000040_793242
link
set_flatten false
source ../scripts/constraint.sdc
compile

##----------------------Write Outputs--------------------
write -f verilog -output /PATH/work/ARA/output/ara_netlist.v
write -format ddc -output /PATH/work/ARA/output/ara.ddc

##----------------------Write Reports--------------------
report_timing -significant_digits 4 -max_path 5 > /PATH/work/ARA/report/ARA_timing.rpt
report_area -hierarchy > /PATH/work/ARA/report/ARA_area.rpt
report_power -hierarchy > /PATH/work/ARA/report/ARA_power.rpt
report_qor > /PATH/work/ARA/report/ARA_qor.rpt
report_design > /PATH/work/ARA/report/ARA_design.rpt

exit
