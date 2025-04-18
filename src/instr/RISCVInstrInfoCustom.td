// simm12_flag: 12-bit signed immediate where imm[11] encodes a flag (weight/activation).
def simm12_flag : Operand<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<12>;
  let EncoderMethod = "getImmOpValue";
  let MIOperandInfo = (ops simm12);
  let PrintMethod = "printSImm12FlagOperand";
}

// U-Type指令模板 (mpcfg)
class RVInstMPU<dag outs, dag ins, string opcodestr, string argstr>
    : RVInstU<OPC_CUSTOM_1, outs, ins, opcodestr, argstr> {
  bits<20> imm20;
  let Inst{31-12} = imm20;    // 20-bit立即数
  let Inst{11-7}  = 0;        // rd=0
  let Inst{6-2}   = 0x0A;     // 特殊功能码
  let Inst{1-0}   = 0x3;      // opcode固定后缀
}

class RVInstMPLE<bits<3> funct3, dag outs, dag ins, 
                string opcodestr, string argstr>
    : RVInstIBase<funct3, OPC_CUSTOM_0, outs, ins, opcodestr, argstr> {
  
  // 字段定义
  let Inst{14-12} = 0b010;   // funct3=6
  let Inst{11-7}  = 0;       // rd
  let Inst{6-2}   = 0x02;    // 功能码
  let Inst{1-0}   = 0x3;     // opcode
  let rd = 0;        // 目标寄存器固定为0
}

class RVInstMPSE<bits<3> funct3, dag outs, dag ins, 
               string opcodestr, string argstr>
    : RVInstIBase<funct3, OPC_CUSTOM_0, outs, ins, opcodestr, argstr> {
  bits<12> imm12;
  let Inst{31-20} = imm12;  // 12-bit立即数
  let Inst{14-12} = 0b100;   // funct3=6
  let Inst{6-2}   = 0x02;     // 功能码基础部分
  let Inst{1-0}   = 0x3;      // opcode固定后缀
  let Inst{11-7}  = 0;        // rd=0
  let rd = 0;        // 目标寄存器固定为0
}


class RVInstMPMM<dag outs, dag ins>
  : RVInst<outs, ins, "", "", [], InstFormatI> {
  bits<7> opcode;
  bits<3> funct3;

  bits<12> imm12;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-20} = 0;
  let Inst{19-15} = 0;
  let Inst{14-12} = 0b110;
  let Inst{11-7} = 0;
  let Inst{6-0} = 0b0001011;
}


// 指令定义
//----------------------------------------------------------


let hasSideEffects = 0 in {
    // mpcfg指令 (U-Type)
    def MCFG : RVInstMPU<(outs), (ins uimm20:$imm20), "mpcfg", "$imm20">, Sched<[WriteIALU]> {
        let mayLoad = 0;
        let mayStore = 0;
    }

    // mple指令 (I-Type)
    def MPLE : RVInstMPLE<0b010, (outs), (ins GPRMem:$rs1, simm12_flag:$imm, uimm1:$isWeight),
                          "mple", "${imm}(${rs1}), $isWeight"> {
      // 寄存器字段位置
      let rd = 0;    // 目标寄存器固定为0
      
      // 内存访问属性
      let mayLoad = 1;
      let mayStore = 0;

      bit isWeight;
      bits<11> imm;
      let Inst{31} = isWeight; // 手动分配最高位
      let Inst{30-20} = imm;   // 手动分配低11位
      let DecoderMethod = "decodeMPLE"; // 指定解码函数
    }

    // mpse指令 (I-Type)
    def MPSE : RVInstMPSE<0b100, (outs), (ins GPRMem:$rs1, simm12:$imm12),
                        "mpse", "${imm12}(${rs1})"> {
        let rd = 0;
        let mayLoad = 0;
        let mayStore = 1;
        let DecoderMethod = "decodeMPSE"; // 指定解码函数
    }

    // // mpmm伪指令定义
    def MPMM : RVInstMPMM<(outs), (ins)> {
        let mayLoad = 0;
        let mayStore = 0;
        let AsmString = "mpmm"; // 汇编格式
    }
    
}

// 给 mpse 指令添加反汇编别名
def : InstAlias<"mpse ${imm12}(${rs1})", (MPSE GPR:$rs1, simm12:$imm12)>;
