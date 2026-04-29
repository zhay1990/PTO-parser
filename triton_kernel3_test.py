import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_3(
    #pypto.language.Tensor[[1, 8192],pypto.language.FP32]
    attn_row_1_ptr, attn_row_1_stride_0, attn_row_1_stride_1, # Output ptr
):
    attn_row_0 = tl.zeros([1, 8192], dtype = tl.float32, )
    attn_row_1 = attn_row_0 * 0
    attn_row_1_offset = (tl.arange(0, 1))[:, None] * attn_row_1_stride_0 + (tl.arange(0, 8192))[None, :] * attn_row_1_stride_1
    tl.store(attn_row_1_ptr + attn_row_1_offset, attn_row_1)

def qwen3_decode_layer_incore_3_torch():
    attn_row_0 = torch.empty([1, 8192], dtype = torch.float32, layout = torch.strided, device='cuda')
    attn_row_1 = torch.mul(attn_row_0, 0)
    return attn_row_1

if __name__ == '__main__':
    attn_row_1 = torch.empty([1, 8192], dtype = torch.float32, device='cuda')

    qwen3_decode_layer_incore_3[(1, )](attn_row_1, attn_row_1.stride(0), attn_row_1.stride(1))
    attn_row_2 = qwen3_decode_layer_incore_3_torch()

    print(torch.allclose(attn_row_1, attn_row_2, atol=1e-3, rtol=1e-3))


    pass