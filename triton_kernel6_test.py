import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_6(
    #pypto.language.Tensor[[16, 8192],pypto.language.FP32]
    attn_out_2_ptr, attn_out_2_stride_0, attn_out_2_stride_1, 
    b0_1,
    #pypto.language.Tensor[[16, 5120],pypto.language.BF16]
    hidden_states_0_ptr, hidden_states_0_stride_0, hidden_states_0_stride_1, 
    #pypto.language.Tensor[[4, 5120],pypto.language.FP32]
    resid1_tile_3_ptr, resid1_tile_3_stride_0, resid1_tile_3_stride_1, 
    #pypto.language.Tensor[[5120, 5120],pypto.language.BF16]
    wo_0_ptr, wo_0_stride_0, wo_0_stride_1, 
):
    ob_4_out = tl.program_id(axis = 0)
    ob_4_in = tl.program_id(axis = 1)
    o0_0 = ((0 + (((ob_4_out * 8) + ob_4_in) * 1)) * 64)
    o_acc_0 = tl.zeros([4, 64], dtype = tl.float32, )
    o_acc_1 = o_acc_0 * 0
    for kb_13 in range(20):
        k0_19 = (kb_13 * 256)
        _t41_offset = (b0_1 + tl.arange(0, 4))[:, None] * attn_out_2_stride_0 + (k0_19 + tl.arange(0, 256))[None, :] * attn_out_2_stride_1
        _t41 = tl.load(attn_out_2_ptr + _t41_offset)
        a_chunk_0 = _t41.to(tl.bfloat16)
        w_chunk_0_offset = (k0_19 + tl.arange(0, 256))[:, None] * wo_0_stride_0 + (o0_0 + tl.arange(0, 64))[None, :] * wo_0_stride_1
        w_chunk_0 = tl.load(wo_0_ptr + w_chunk_0_offset)
        _t42 = tl.dot(a_chunk_0, w_chunk_0)
        o_acc_1 = o_acc_1 + _t42
    _t43_offset = (b0_1 + tl.arange(0, 4))[:, None] * hidden_states_0_stride_0 + (o0_0 + tl.arange(0, 64))[None, :] * hidden_states_0_stride_1
    _t43 = tl.load(hidden_states_0_ptr + _t43_offset)
    resid_0 = _t43.to(tl.float32)
    _t44 = o_acc_1 + resid_0
    resid1_tile_3_offset = (0 + tl.arange(0, 4))[:, None] * resid1_tile_3_stride_0 + (o0_0 + tl.arange(0, 64))[None, :] * resid1_tile_3_stride_1
    tl.store(resid1_tile_3_ptr + resid1_tile_3_offset, _t44)

def qwen3_decode_layer_incore_6_torch(attn_out_2, b0_1, hidden_states_0, ob_4_out, resid1_tile_3, wo_0):
    for ob_4_in in range(8):
        o0_0 = ((0 + (((ob_4_out * 8) + ob_4_in) * 1)) * 64)
        o_acc_0 = torch.empty([4, 64], dtype = torch.float32, layout = torch.strided, device='cuda')
        o_acc_1 = torch.mul(o_acc_0, 0)
        for kb_13 in range(20):
            k0_19 = (kb_13 * 256)
            _t41 = attn_out_2[b0_1 : b0_1 + 4, k0_19 : k0_19 + 256]
            a_chunk_0 = _t41.to(torch.bfloat16)
            w_chunk_0 = wo_0[k0_19 : k0_19 + 256, o0_0 : o0_0 + 64]
            _t42 = torch.matmul(a_chunk_0, w_chunk_0)
            o_acc_1 = torch.add(o_acc_1, _t42)
        _t43 = hidden_states_0[b0_1 : b0_1 + 4, o0_0 : o0_0 + 64]
        resid_0 = _t43.to(torch.float32)
        _t44 = torch.add(o_acc_1, resid_0)
        resid1_tile_3 [0: 0 + _t44.shape[0], o0_0: o0_0 + _t44.shape[1]] = _t44
    return resid1_tile_3

if __name__ == '__main__':
    attn_out_0 = torch.rand([16, 8192], dtype = torch.float32, device = 'cuda')
    hidden_states_0 = torch.rand([16, 5120], dtype = torch.bfloat16, device='cuda')
    resid1_tile_3 = torch.empty([4, 5120], dtype = torch.float32, device = 'cuda')
    resid1_tile_3_torch = torch.empty_like(resid1_tile_3)
    wo_0 = torch.rand([5120, 5120], dtype = torch.bfloat16, device='cuda')

    for b0_1 in range(0, 16, 4,):
        qwen3_decode_layer_incore_6[(10, 8, )](attn_out_0, attn_out_0.stride(0), attn_out_0.stride(1), b0_1, hidden_states_0, hidden_states_0.stride(0), hidden_states_0.stride(1), resid1_tile_3, resid1_tile_3.stride(0), resid1_tile_3.stride(1), wo_0, wo_0.stride(0), wo_0.stride(1), )
        for ob_4_out in range(10):
            resid1_tile_3_torch = qwen3_decode_layer_incore_6_torch(attn_out_0, b0_1, hidden_states_0, ob_4_out, resid1_tile_3_torch, wo_0)

    print(torch.allclose(resid1_tile_3, resid1_tile_3_torch, atol=1e-2, rtol=1e-2))


    pass