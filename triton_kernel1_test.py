import os
os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_1(
    b0_0,
    #pypto.language.Tensor[[16, 5120],pypto.language.BF16]
    hidden_states_0_ptr, hidden_states_0_stride_0, hidden_states_0_stride_1, 
    #pypto.language.Tensor[[1, 5120],pypto.language.FP32]
    input_rms_weight_0_ptr, input_rms_weight_0_stride_0, input_rms_weight_0_stride_1, 
    #pypto.language.Tensor[[4, 1],pypto.language.FP32]
    inv_rms_tile_0_ptr, inv_rms_tile_0_stride_0, inv_rms_tile_0_stride_1, 
    #pypto.language.Tensor[[16, 8192],pypto.language.BF16]
    q_proj_5_ptr, q_proj_5_stride_0, q_proj_5_stride_1, 
    #pypto.language.Tensor[[5120, 5120],pypto.language.BF16]
    wq_0_ptr, wq_0_stride_0, wq_0_stride_1, 
):
    ob_0_out = tl.program_id(axis = 0)
    ob_0_in = tl.program_id(axis = 1)
    inv_rms_tile_0 = tl.load(inv_rms_tile_0_ptr)
    q0_0 = ((0 + (((ob_0_out * 4) + ob_0_in) * 1)) * 64)
    q_acc_0 = tl.zeros([4, 64], dtype = tl.float32, )
    q_acc_1 = q_acc_0 * 0
    for kb_5 in range(20):
        k0_7 = (kb_5 * 256)
        x_chunk_bf16_0_offset = (b0_0 + tl.arange(0, 4))[:, None] * hidden_states_0_stride_0 + (k0_7 + tl.arange(0, 256))[None, :] * hidden_states_0_stride_1
        x_chunk_bf16_0 = tl.load(hidden_states_0_ptr + x_chunk_bf16_0_offset)
        x_chunk_7 = x_chunk_bf16_0.to(tl.float32)
        gamma_0_offset = (0 + tl.arange(0, 1))[:, None] * input_rms_weight_0_stride_0 + (k0_7 + tl.arange(0, 256))[None, :] * input_rms_weight_0_stride_1
        gamma_0 = tl.load(input_rms_weight_0_ptr + gamma_0_offset)
        _t5 = x_chunk_7 * inv_rms_tile_0
        normed_0 = _t5 * gamma_0
        wq_chunk_0_offset = (k0_7 + tl.arange(0, 256))[:, None] * wq_0_stride_0 + (q0_0 + tl.arange(0, 64))[None, :] * wq_0_stride_1
        wq_chunk_0 = tl.load(wq_0_ptr + wq_chunk_0_offset)
        _t6 = normed_0.to(tl.bfloat16)
        _t7 = tl.dot(_t6, wq_chunk_0)
        q_acc_1 = q_acc_1 + _t7
    _t8 = q_acc_1.to(tl.bfloat16)
    q_proj_5_offset = (b0_0 + tl.arange(0, 4))[:, None] * q_proj_5_stride_0 + (q0_0 + tl.arange(0, 64))[None, :] * q_proj_5_stride_1
    tl.store(q_proj_5_ptr + q_proj_5_offset, _t8)

def qwen3_decode_layer_incore_1_torch(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, ob_0_out, q_proj_5, wq_0):
    for ob_0_in in range(4):
        q0_0 = ((0 + (((ob_0_out * 4) + ob_0_in) * 1)) * 64)
        q_acc_0 = torch.empty([4, 64], dtype = torch.float32, layout = torch.strided)
        q_acc_1 = torch.mul(q_acc_0, 0)
        for kb_5 in range(20):
            k0_7 = (kb_5 * 256)
            x_chunk_bf16_0 = hidden_states_0[b0_0 : b0_0 + 4, k0_7 : k0_7 + 256]
            x_chunk_7 = x_chunk_bf16_0.to(torch.float32)
            gamma_0 = input_rms_weight_0[0 : 0 + 1, k0_7 : k0_7 + 256]
            _t5 = x_chunk_7 * inv_rms_tile_0
            normed_0 = _t5 * gamma_0
            wq_chunk_0 = wq_0[k0_7 : k0_7 + 256, q0_0 : q0_0 + 64]
            _t6 = normed_0.to(torch.bfloat16)
            _t7 = torch.matmul(_t6, wq_chunk_0)
            q_acc_1 = torch.add(q_acc_1, _t7)
        _t8 = q_acc_1.to(torch.bfloat16)
        q_proj_5 [b0_0: b0_0 + _t8.shape[0], q0_0: q0_0 + _t8.shape[1]] = _t8
    return q_proj_5

if __name__ == '__main__':
    hidden_states_0 = torch.rand([16, 5120], dtype = torch.bfloat16)
    input_rms_weight_0 = torch.rand([1, 5120], dtype = torch.float32)
    inv_rms_0 = torch.rand([16, 1], dtype = torch.float32)
    q_proj_0 = torch.empty([16, 8192], dtype = torch.bfloat16, layout = torch.strided)
    q_proj_1 = q_proj_0
    wq_0 = torch.rand([5120, 5120], dtype = torch.bfloat16)
    
    for b0_0 in range(0, 16, 4, ):
        inv_rms_tile_0 = inv_rms_0[b0_0 : b0_0 + 4, 0 : 0 + 1]
        qwen3_decode_layer_incore_1[(20, 4, )](b0_0, hidden_states_0, hidden_states_0.stride(0), hidden_states_0.stride(1), input_rms_weight_0, input_rms_weight_0.stride(0), input_rms_weight_0.stride(1), inv_rms_tile_0, inv_rms_tile_0.stride(0), inv_rms_tile_0.stride(1), q_proj_0, q_proj_0.stride(0), q_proj_0.stride(1), wq_0, wq_0.stride(0), wq_0.stride(1), )

    for b0_0 in range(0, 16, 4, ):
        inv_rms_tile_0 = inv_rms_0[b0_0 : b0_0 + 4, 0 : 0 + 1]
        for ob_0_out in range(20):
            q_proj = qwen3_decode_layer_incore_1_torch(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, ob_0_out, q_proj_1, wq_0)

    print(torch.allclose(q_proj_0, q_proj, atol=1e-3, rtol=1e-3))

    pass