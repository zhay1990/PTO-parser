import importlib
import torch
# 使用字符串路径动态导入
module = importlib.import_module("out.07_correct_torch")

if __name__ == '__main__':
    test = module.Qwen3SingleLayerDecode_torch()
    hidden_states = torch.rand([16, 5120], dtype = torch.bfloat16)
    seq_lens = torch.randint(1, 4096, [16], dtype = torch.int32)
    rope_cos = torch.rand([4096, 128], dtype = torch.float32)
    rope_sin = torch.rand([4096, 128], dtype = torch.float32)
    k_cache = torch.rand([524288, 128], dtype = torch.bfloat16)
    
    v_cache = torch.rand([524288, 128], dtype = torch.bfloat16)
    input_rms_weight = torch.rand([1, 5120], dtype = torch.float32)
    wq = torch.rand([5120, 5120], dtype = torch.bfloat16)
    wk = torch.rand([5120, 1024], dtype = torch.bfloat16)
    wv = torch.rand([5120, 1024], dtype = torch.bfloat16)
    wo = torch.rand([5120, 5120], dtype = torch.bfloat16)
    post_rms_weight = torch.rand([1, 5120], dtype = torch.float32)
    w_gate = torch.rand([5120, 25600], dtype = torch.bfloat16)
    w_up = torch.rand([5120, 25600], dtype = torch.bfloat16)
    w_down = torch.rand([25600, 5120], dtype = torch.bfloat16)
    out = torch.empty([16, 5120], dtype = torch.float32)
    
    test.qwen3_decode_layer(hidden_states, seq_lens, rope_cos, rope_sin, k_cache, v_cache, input_rms_weight, wq, wk, wv, wo, post_rms_weight, w_gate, w_up, w_down, out)

    print("Complete!")