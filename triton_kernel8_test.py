import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_8(
    b0_1,
    #pypto.language.Tensor[[4, 5120],pypto.language.FP32]
    down_proj_tile_3_ptr, down_proj_tile_3_stride_0, down_proj_tile_3_stride_1, 
    #pypto.language.Tensor[[16, 5120],pypto.language.BF16]
    out_5_ptr, out_5_stride_0, out_5_stride_1, 
    #pypto.language.Tensor[[4, 5120],pypto.language.FP32]
    resid1_tile_iter_1_outer_l0_rv_ptr, resid1_tile_iter_1_outer_l0_rv_stride_0, resid1_tile_iter_1_outer_l0_rv_stride_1, 
):
    ob_6_out = tl.program_id(axis = 0)
    ob_6_in = tl.program_id(axis = 1)
    o0_6 = ((0 + (((ob_6_out * 4) + ob_6_in) * 1)) * 64)
    # 因为输入的PTO源码没有边界检查, 所以没有为这个tl.load生成Mask
    _t58_offset = (0 + tl.arange(0, 4))[:, None] * down_proj_tile_3_stride_0 + (o0_6 + tl.arange(0, 64))[None, :] * down_proj_tile_3_stride_1
    _t58 = tl.load(down_proj_tile_3_ptr + _t58_offset)
    # 因为输入的PTO源码没有边界检查, 所以没有为这个tl.load生成Mask
    _t59_offset = (0 + tl.arange(0, 4))[:, None] * resid1_tile_iter_1_outer_l0_rv_stride_0 + (o0_6 + tl.arange(0, 64))[None, :] * resid1_tile_iter_1_outer_l0_rv_stride_1
    _t59 = tl.load(resid1_tile_iter_1_outer_l0_rv_ptr + _t59_offset)
    down_acc_0 = _t58 + _t59
    _t60 = down_acc_0.to(tl.bfloat16)
    out_5_offset = (b0_1 + tl.arange(0, 4))[:, None] * out_5_stride_0 + (o0_6 + tl.arange(0, 64))[None, :] * out_5_stride_1
    tl.store(out_5_ptr + out_5_offset, _t60)

def qwen3_decode_layer_incore_8_torch(b0_1, down_proj_tile_3, ob_6_out, out_5, resid1_tile_iter_1_outer_l0_rv):
    for ob_6_in in range(4):
        o0_6 = ((0 + (((ob_6_out * 4) + ob_6_in) * 1)) * 64)
        _t58 = down_proj_tile_3[0 : 0 + 4, o0_6 : o0_6 + 64]
        _t59 = resid1_tile_iter_1_outer_l0_rv[0 : 0 + 4, o0_6 : o0_6 + 64]
        down_acc_0 = torch.add(_t58, _t59)
        _t60 = down_acc_0.to(torch.bfloat16)
        out_5 [b0_1: b0_1 + _t60.shape[0], o0_6: o0_6 + _t60.shape[1]] = _t60
    return out_5
    
if __name__ == '__main__':
    down_proj_tile_0 = torch.rand([4, 5120], dtype=torch.float32, device='cuda')
    ret_4_2____ = torch.rand([4, 5120], dtype = torch.float32, layout = torch.strided, device = 'cuda')
    out_0 = torch.empty([16, 5120], dtype = torch.bfloat16, device='cuda')
    out_5 = torch.empty_like(out_0)
    for b0_1 in range(0, 16, 4, ):
        qwen3_decode_layer_incore_8[(20, 4, )](b0_1, down_proj_tile_0, down_proj_tile_0.stride(0), down_proj_tile_0.stride(1), out_0, out_0.stride(0), out_0.stride(1), ret_4_2____, ret_4_2____.stride(0), ret_4_2____.stride(1), )
        for ob_6_out in range(20):
            out_5 = qwen3_decode_layer_incore_8_torch(b0_1, down_proj_tile_0, ob_6_out, out_5, ret_4_2____)
    
    print(torch.allclose(out_0, out_5, atol=1e-3, rtol=1e-3))


    pass