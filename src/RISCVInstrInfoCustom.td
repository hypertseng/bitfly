// U-Type指令模板 (mpcfg)
class RVInstMPU<dag outs, dag ins, string opcodestr, string argstr>
    : RVInstU<OPC_CUSTOM_1, outs, ins, opcodestr, argstr> {
  bits<20> imm20;
  let Inst{31-12} = imm20;    // 20-bit立即数
  let Inst{11-7}  = 0;        // rd=0
  let Inst{6-2}   = 0x0A;     // 特殊功能码
  let Inst{1-0}   = 0x3;      // opcode固定后缀
}

// I-Type指令模板 (mple/mpse)
class RVInstMPI<bits<3> funct3, dag outs, dag ins, 
               string opcodestr, string argstr>
    : RVInstI<funct3, OPC_CUSTOM_0, outs, ins, opcodestr, argstr> {
  let Inst{6-2}   = 0x02;     // 功能码基础部分
  let Inst{1-0}   = 0x3;      // opcode固定后缀
  let Inst{11-7}  = 0;        // rd=0
}


// 伪指令定义模板
class MPMM_Pseudo<dag outs, dag ins, string opcodestr, string argstr>
    : Pseudo<outs, ins, [], opcodestr, argstr> {
  
  // 编码固定值
  let Inst{31-15} = 0;       // 高位全零
  let Inst{14-12} = 0b110;   // funct3=6
  let Inst{11-7}  = 0;       // rd=0
  let Inst{6-2}   = 0x02;    // 功能码 (0b00010)
  let Inst{1-0}   = 0x3;     // opcode后缀
  
}

// 指令定义
//----------------------------------------------------------

let hasSideEffects = 0 in {
    // mpcfg指令 (U-Type)
    def MCFG : RVInstMPU<(outs), (ins uimm20:$imm20), "mpcfg", "$imm20"> {
        let mayLoad = 0;
        let mayStore = 0;
    }

    // mple指令 (I-Type)
    def MPLE : RVInstMPI<0b010, (outs), (ins GPR:$rs1, simm12:$imm12),
                        "mple", "$rs1, $imm12"> {
        let rs1 = 1;  // rs1字段位置
        let mayLoad = 1;
        let mayStore = 0;
    }

    // mpse指令 (I-Type)
    def MPSE : RVInstMPI<0b100, (outs), (ins GPR:$rs1, simm12:$imm12),
                        "mpse", "$rs1, $imm12"> {
        let rs1 = 1;
        let mayLoad = 0;
        let mayStore = 1;
    }

    // mpmm伪指令定义
    def MPMM : MPMM_Pseudo<(outs), (ins), "mpmm", ""> {
        let AsmString = "";  // 汇编格式为空
        let mayLoad = 0;
        let mayStore = 0;
    }
}
