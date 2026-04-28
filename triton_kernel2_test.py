import os
os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_2(
    b0_0,
    #pypto.language.Tensor[[16, 5120],pypto.language.BF16]
    hidden_states_0_ptr, hidden_states_0_stride_0, hidden_states_0_stride_1, 
    #pypto.language.Tensor[[1, 5120],pypto.language.FP32]
    input_rms_weight_0_ptr, input_rms_weight_0_stride_0, input_rms_weight_0_stride_1, 
    #pypto.language.Tensor[[4, 1],pypto.language.FP32]
    inv_rms_tile_0_ptr, inv_rms_tile_0_stride_0, inv_rms_tile_0_stride_1, 
    #pypto.language.Tensor[[16, 1024],pypto.language.BF16]
    k_proj_5_ptr, k_proj_5_stride_0, k_proj_5_stride_1, 
    #pypto.language.Tensor[[16, 1024],pypto.language.BF16]
    v_proj_5_ptr, v_proj_5_stride_0, v_proj_5_stride_1, 
    #pypto.language.Tensor[[5120, 1024],pypto.language.BF16]
    wk_0_ptr, wk_0_stride_0, wk_0_stride_1, 
    #pypto.language.Tensor[[5120, 1024],pypto.language.BF16]
    wv_0_ptr, wv_0_stride_0, wv_0_stride_1, 
):
    ob_1_out = tl.program_id(axis = 0)
    ob_1_in = tl.program_id(axis = 1)
    inv_rms_offset = (tl.arange(0, 4))[:, None] * inv_rms_tile_0_stride_0
    inv_rms_tile_0 = tl.load(inv_rms_tile_0_ptr + inv_rms_offset)
    kv0_0 = ((0 + (((ob_1_out * 8) + ob_1_in) * 1)) * 32)
    k_acc_0 = tl.zeros([4, 32], dtype = tl.float32, )
    v_acc_0 = tl.zeros([4, 32], dtype = tl.float32, )
    k_acc_1 = k_acc_0 * 0
    v_acc_1 = v_acc_0 * 0
    for kb_8 in range(20):
        k0_12 = (kb_8 * 256)
        x_chunk_bf16_5_offset = (b0_0 + tl.arange(0, 4))[:, None] * hidden_states_0_stride_0 + (k0_12 + tl.arange(0, 256))[None, :] * hidden_states_0_stride_1
        x_chunk_bf16_5 = tl.load(hidden_states_0_ptr + x_chunk_bf16_5_offset)
        x_chunk_12 = x_chunk_bf16_5.to(tl.float32)
        gamma_5_offset = (0 + tl.arange(0, 1))[:, None] * input_rms_weight_0_stride_0 + (k0_12 + tl.arange(0, 256))[None, :] * input_rms_weight_0_stride_1
        gamma_5 = tl.load(input_rms_weight_0_ptr + gamma_5_offset)
        _t9 = x_chunk_12 * inv_rms_tile_0
        normed_5 = _t9 * gamma_5
        normed_bf16_0 = normed_5.to(tl.bfloat16)
        wk_chunk_0_offset = (k0_12 + tl.arange(0, 256))[:, None] * wk_0_stride_0 + (kv0_0 + tl.arange(0, 32))[None, :] * wk_0_stride_1
        wk_chunk_0 = tl.load(wk_0_ptr + wk_chunk_0_offset)
        wv_chunk_0_offset = (k0_12 + tl.arange(0, 256))[:, None] * wv_0_stride_0 + (kv0_0 + tl.arange(0, 32))[None, :] * wv_0_stride_1
        wv_chunk_0 = tl.load(wv_0_ptr + wv_chunk_0_offset)
        _t10 = tl.dot(normed_bf16_0, wk_chunk_0)
        k_acc_1 = k_acc_1 + _t10
        _t11 = tl.dot(normed_bf16_0, wv_chunk_0)
        v_acc_1 = v_acc_1 + _t11
    _t12 = k_acc_1.to(tl.bfloat16)
    k_proj_5_offset = (b0_0 + tl.arange(0, 4))[:, None] * k_proj_5_stride_0 + (kv0_0 + tl.arange(0, 32))[None, :] * k_proj_5_stride_1
    tl.store(k_proj_5_ptr + k_proj_5_offset, _t12)
    _t13 = v_acc_1.to(tl.bfloat16)
    v_proj_5_offset = (b0_0 + tl.arange(0, 4))[:, None] * v_proj_5_stride_0 + (kv0_0 + tl.arange(0, 32))[None, :] * v_proj_5_stride_1
    tl.store(v_proj_5_ptr + v_proj_5_offset, _t13)

def qwen3_decode_layer_incore_2_torch(self, b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k_proj_5, ob_1_out, v_proj_5, wk_0, wv_0):
    for ob_1_in in range(8):
        kv0_0 = ((0 + (((ob_1_out * 8) + ob_1_in) * 1)) * 32)
        k_acc_0 = torch.empty([4, 32], dtype = torch.float32, layout = torch.strided)
        v_acc_0 = torch.empty([4, 32], dtype = torch.float32, layout = torch.strided)
        k_acc_1 = torch.mul(k_acc_0, 0)
        v_acc_1 = torch.mul(v_acc_0, 0)
        for kb_8 in range(20):
            k0_12 = (kb_8 * 256)
            x_chunk_bf16_5 = hidden_states_0[b0_0 : b0_0 + 4, k0_12 : k0_12 + 256]
            x_chunk_12 = x_chunk_bf16_5.to(torch.float32)
            gamma_5 = input_rms_weight_0[0 : 0 + 1, k0_12 : k0_12 + 256]
            _t9 = x_chunk_12 * inv_rms_tile_0
            normed_5 = _t9 * gamma_5
            normed_bf16_0 = normed_5.to(torch.bfloat16)
            wk_chunk_0 = wk_0[k0_12 : k0_12 + 256, kv0_0 : kv0_0 + 32]
            wv_chunk_0 = wv_0[k0_12 : k0_12 + 256, kv0_0 : kv0_0 + 32]
            _t10 = torch.matmul(normed_bf16_0, wk_chunk_0)
            k_acc_1 = torch.add(k_acc_1, _t10)
            _t11 = torch.matmul(normed_bf16_0, wv_chunk_0)
            v_acc_1 = torch.add(v_acc_1, _t11)
        _t12 = k_acc_1.to(torch.bfloat16)
        k_proj_5 [b0_0: b0_0 + _t12.shape[0], kv0_0: kv0_0 + _t12.shape[1]] = _t12
        _t13 = v_acc_1.to(torch.bfloat16)
        v_proj_5 [b0_0: b0_0 + _t13.shape[0], kv0_0: kv0_0 + _t13.shape[1]] = _t13
    return k_proj_5, v_proj_5

if __name__ == '__main__':
    hidden_states_0 = torch.rand([16, 5120], dtype = torch.bfloat16)
    input_rms_weight_0 = torch.rand([1, 5120], dtype = torch.float32)
    inv_rms_0 = torch.rand([4, 1], dtype = torch.float32)
    
    k_proj_0 = torch.empty([16, 1024], dtype = torch.bfloat16, layout = torch.strided)
    v_proj_0 = torch.empty([16, 1024], dtype = torch.bfloat16, layout = torch.strided)
    k_proj_1 = torch.empty_like(k_proj_0)
    v_proj_1 = torch.empty_like(v_proj_0)

    wq_0 = torch.rand([5120, 5120], dtype = torch.bfloat16)
    wk_0 = torch.rand([5120, 1024], dtype = torch.bfloat16)
    wv_0 = torch.rand([5120, 1024], dtype = torch.bfloat16)

    for b0_0 in range(0, 16, 4, ):
        inv_rms_tile_0 = inv_rms_0[b0_0 : b0_0 + 4, 0 : 0 + 1]
        qwen3_decode_layer_incore_2[(4, 8, )](b0_0, hidden_states_0, hidden_states_0.stride(0), hidden_states_0.stride(1), input_rms_weight_0, input_rms_weight_0.stride(0), input_rms_weight_0.stride(1), inv_rms_tile_0, inv_rms_tile_0.stride(0), inv_rms_tile_0.stride(1), k_proj_0, k_proj_0.stride(0), k_proj_0.stride(1), v_proj_0, v_proj_0.stride(0), v_proj_0.stride(1), wk_0, wk_0.stride(0), wk_0.stride(1), wv_0, wv_0.stride(0), wv_0.stride(1), )

    for b0_0 in range(0, 16, 4):
        inv_rms_tile_0 = inv_rms_0[b0_0 : b0_0 + 4, 0 : 0 + 1]
        for ob_1_out in range(4):
            (k_proj_0, v_proj_0, ) = self.qwen3_decode_layer_incore_2_torch(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k_proj_1, ob_1_out, v_proj_1, wk_0, wv_0)

    print(torch.allclose(k_proj_0, k_proj, atol=1e-3, rtol=1e-3))
    print(torch.allclose(v_proj_0, v_proj, atol=1e-3, rtol=1e-3))


    pass