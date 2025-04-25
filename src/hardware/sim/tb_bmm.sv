initial begin
  // 触发 SA 操作
  pe_req_i.op = SA_OP;
  pe_req_valid_i = 1;

  // 写入 SA 输入队列
  for (int i = 0; i < 4; i++) begin
    operand_i[MPUAct0+i] = 64'h1234 + i;
    operand_valid_i[MPUAct0+i] = 1;
  end

  // 等待 SA 输出
  wait (sa_output_valid == 4'b1111);
  foreach (sa_output_data[i]) begin
    assert (sa_output_data[i] == expected_value[i])
    else $error("SA output mismatch!");
  end
end
