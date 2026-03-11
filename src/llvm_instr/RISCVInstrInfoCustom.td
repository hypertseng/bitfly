// simm12_flag: 12-bit signed immediate where imm[11] encodes a flag (weight/activation).
def simm12_flag : Operand<XLenVT> {
  let ParserMatchClass = SImmAsmOperand<12>;
  let EncoderMethod = "getImmOpValue";
  let MIOperandInfo = (ops simm12);
  let PrintMethod = "printSImm12FlagOperand";
}

def BMPCFGImmAsmOperand : AsmOperandClass {
  let Name = "BMPCFGImm";
  let ParserMethod = "parseBMPCFGImm";
  let RenderMethod = "addBMPCFGImmOperands";
  let PredicateMethod = "isBMPCFGImm";
  let DiagnosticType = "InvalidBMPCFGImm";
}

def bmpcfg_imm25 : Operand<XLenVT> {
  let ParserMatchClass = BMPCFGImmAsmOperand;
  let EncoderMethod = "getBMPCFGImm25OpValue";
  let DecoderMethod = "decodeBMPCFGImm25";
  let PrintMethod = "printBMPCFGImm25";
}

class RVInstBMPU_CFG<dag outs, dag ins, string opcodestr, string argstr>
    : RVInstU<OPC_CUSTOM_1, outs, ins, opcodestr, argstr> {

  bits<25> cfg25;

  let Inst{31-12} = cfg25{24-5};

  let Inst{11-7}  = cfg25{4-0};
  let Inst{6-2}   = 0x0A;
  let Inst{1-0}   = 0x3;
}
class RVInstBMPLE<bits<3> funct3, dag outs, dag ins, 
                string opcodestr, string argstr>
    : RVInstIBase<funct3, OPC_CUSTOM_0, outs, ins, opcodestr, argstr> {
  
  // 字段定义
  let Inst{14-12} = 0b010;   // funct3=6
  let Inst{11-7}  = 0;       // rd
  let Inst{6-2}   = 0x02;    // 功能码
  let Inst{1-0}   = 0x3;     // opcode
  let rd = 0;        // 目标寄存器固定为0
}

class RVInstBMPSE<bits<3> funct3, dag outs, dag ins, 
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


class RVInstBMPMM<dag outs, dag ins>
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
    // bmpcfg指令 (U-Type)
    def BMPCFG : RVInstBMPU_CFG<
        (outs),
        (ins bmpcfg_imm25:$cfg),
        "bmpcfg",
        "$cfg"
    >, Sched<[WriteIALU]> {
      let mayLoad = 0;
      let mayStore = 0;

      bits<25> cfg;
      let cfg25 = cfg;
    }

    // bmple指令 (I-Type)
    def BMPLE : RVInstBMPLE<0b010, (outs), (ins GPRMem:$rs1, simm12_flag:$imm, uimm1:$isWeight),
                          "bmple", "${imm}(${rs1}), $isWeight"> {
      // 寄存器字段位置
      let rd = 0;    // 目标寄存器固定为0
      
      // 内存访问属性
      let mayLoad = 1;
      let mayStore = 0;

      bit isWeight;
      bits<11> imm;
      let Inst{31} = isWeight; // 手动分配最高位
      let Inst{30-20} = imm;   // 手动分配低11位
      let DecoderMethod = "decodeBMPLE"; // 指定解码函数
    }

    // bmpse指令 (I-Type)
    def BMPSE : RVInstBMPSE<0b100, (outs), (ins GPRMem:$rs1, simm12:$imm12),
                        "bmpse", "${imm12}(${rs1})"> {
        let rd = 0;
        let mayLoad = 0;
        let mayStore = 1;
        let DecoderMethod = "decodeBMPSE"; // 指定解码函数
    }

    // // bmpmm伪指令定义
    def BMPMM : RVInstBMPMM<(outs), (ins)> {
        let mayLoad = 0;
        let mayStore = 0;
        let AsmString = "bmpmm"; // 汇编格式
    }
    
}

// 给 bmpse 指令添加反汇编别名
def : InstAlias<"bmpse ${imm12}(${rs1})", (BMPSE GPR:$rs1, simm12:$imm12)>;
