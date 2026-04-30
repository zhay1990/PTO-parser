import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_5(
    #pypto.language.Tensor[[16, 8192],pypto.language.FP32]
    attn_out_iter_1_ptr, attn_out_iter_1_stride_0, attn_out_iter_1_stride_1, 
    #pypto.language.Tensor[[1, 8192],pypto.language.FP32]
    attn_row_iter_2_outer_l0_rv_ptr, attn_row_iter_2_outer_l0_rv_stride_0, attn_row_iter_2_outer_l0_rv_stride_1, 
    b_0,
    #pypto.language.Tensor[[16, 8192],pypto.language.FP32]
    attn_out_3_ptr, attn_out_3_stride_0, attn_out_3_stride_1, # Output ptr
):
    # 实际尺寸和原始尺寸一致, 不需要mask
    attn_out_iter_1_offset = (tl.arange(0, 16))[:, None] * attn_out_iter_1_stride_0 + (tl.arange(0, 8192))[None, :] * attn_out_iter_1_stride_1
    attn_out_iter_1 = tl.load(attn_out_iter_1_ptr + attn_out_iter_1_offset)
    # 实际尺寸和原始尺寸一致, 不需要mask
    attn_row_iter_2_outer_l0_rv_offset = (tl.arange(0, 1))[:, None] * attn_row_iter_2_outer_l0_rv_stride_0 + (tl.arange(0, 8192))[None, :] * attn_row_iter_2_outer_l0_rv_stride_1
    attn_row_iter_2_outer_l0_rv = tl.load(attn_row_iter_2_outer_l0_rv_ptr + attn_row_iter_2_outer_l0_rv_offset)
    attn_out_iter_1_offset_ = (tl.arange(0, 16))[:, None] * attn_out_iter_1_stride_0 + (tl.arange(0, 8192))[None, :] * attn_out_iter_1_stride_1
    tl.store(attn_out_3_ptr + attn_out_iter_1_offset_, attn_out_iter_1)
    attn_out_3_offset = (b_0 + tl.arange(0, 1))[:, None] * attn_out_3_stride_0 + (0 + tl.arange(0, 8192))[None, :] * attn_out_3_stride_1
    tl.store(attn_out_3_ptr + attn_out_3_offset, attn_row_iter_2_outer_l0_rv)

def qwen3_decode_layer_incore_5_torch(attn_out_iter_1, attn_row_iter_2_outer_l0_rv, b_0):
    attn_out_3 = attn_out_iter_1
    attn_out_3 [b_0: b_0 + attn_row_iter_2_outer_l0_rv.shape[0], 0: 0 + attn_row_iter_2_outer_l0_rv.shape[1]] = attn_row_iter_2_outer_l0_rv
    return attn_out_3

if __name__ == '__main__':
    attn_out_1 = torch.empty([16, 8192], dtype=torch.float32, device='cuda')
    attn_out_1_torch = torch.empty_like(attn_out_1)

    for b_0 in range(16):
        attn_out_2 = torch.rand([1, 8192], dtype=torch.float32, device='cuda')
        qwen3_decode_layer_incore_5[(1, )](attn_out_1, attn_out_1.stride(0), attn_out_1.stride(1), attn_out_2, attn_out_2.stride(0), attn_out_2.stride(1), b_0, attn_out_1, attn_out_1.stride(0), attn_out_1.stride(1))
        attn_out_1_torch = qwen3_decode_layer_incore_5_torch(attn_out_1_torch, attn_out_2, b_0)

    print(torch.allclose(attn_out_1, attn_out_1_torch, atol=1e-3, rtol=1e-3))


    pass