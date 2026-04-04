#-------------------Specify Libraries------------------
set target_library "/PATH/TO/YOUR/PDK"
set link_library "* /PATH/TO/YOUR/PDK"

source /PATH/TO/ara.tcl

set_host_options -max_cores 3

elaborate ara_soc -parameter "NrLanes = 4, VLEN = 4096"
uniquify
current_design ara_00000004_00001000_00000001_38_1_1_1_00000080_00000040_793242
link
set_flatten false
source /PATH/TO/constraint.sdc
compile

##----------------------Write Outputs--------------------
write -f verilog -output /PATH/TO/ara_netlist.v
write -format ddc -output /PATH/TO/ara.ddc

##----------------------Write Reports--------------------
report_timing -significant_digits 4 -max_path 5 > /PATH/TO/ARA_timing.rpt
report_area -hierarchy > /PATH/TO/ARA_area.rpt
report_power -hierarchy > /PATH/TO/ARA_power.rpt
report_qor > /PATH/TO/ARA_qor.rpt
report_design > /PATH/TO/ARA_design.rpt

exit
