import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_7(
    #pypto.language.Tensor[[4, 5120],pypto.language.FP32]
    down_proj_tile_6_ptr, down_proj_tile_6_stride_0, down_proj_tile_6_stride_1, 
    #pypto.language.Tensor[[4, 32],pypto.language.BF16]
    mlp_chunk_bf16_0_ptr, mlp_chunk_bf16_0_stride_0, mlp_chunk_bf16_0_stride_1, 
    o0_3,
    #pypto.language.Tensor[[25600, 5120],pypto.language.BF16]
    w_down_0_ptr, w_down_0_stride_0, w_down_0_stride_1, 
):
    dob_0_out = tl.program_id(axis = 0)
    dob_0_in = tl.program_id(axis = 1)
    mlp_chunk_bf16_0_offset = (tl.arange(0, 4))[:, None] * mlp_chunk_bf16_0_stride_0 + (tl.arange(0, 32))[None, :] * mlp_chunk_bf16_0_stride_1
    mlp_chunk_bf16_0 = tl.load(mlp_chunk_bf16_0_ptr + mlp_chunk_bf16_0_offset)
    d0_0 = ((0 + (((dob_0_out * 4) + dob_0_in) * 1)) * 64)
    down_prev_0_offset = (0 + tl.arange(0, 4))[:, None] * down_proj_tile_6_stride_0 + (d0_0 + tl.arange(0, 64))[None, :] * down_proj_tile_6_stride_1
    down_prev_0 = tl.load(down_proj_tile_6_ptr + down_prev_0_offset)
    w_down_chunk_0_offset = (o0_3 + tl.arange(0, 32))[:, None] * w_down_0_stride_0 + (d0_0 + tl.arange(0, 64))[None, :] * w_down_0_stride_1
    w_down_chunk_0 = tl.load(w_down_0_ptr + w_down_chunk_0_offset)
    _t57 = tl.dot(mlp_chunk_bf16_0, w_down_chunk_0)
    down_next_0 = down_prev_0 + _t57
    down_proj_tile_6_offset = (0 + tl.arange(0, 4))[:, None] * down_proj_tile_6_stride_0 + (d0_0 + tl.arange(0, 64))[None, :] * down_proj_tile_6_stride_1
    tl.store(down_proj_tile_6_ptr + down_proj_tile_6_offset, down_next_0)

def qwen3_decode_layer_incore_7_torch(dob_0_out, down_proj_tile_6, mlp_chunk_bf16_0, o0_3, w_down_0):
    for dob_0_in in range(4):
        d0_0 = ((0 + (((dob_0_out * 4) + dob_0_in) * 1)) * 64)
        down_prev_0 = down_proj_tile_6[0 : 0 + 4, d0_0 : d0_0 + 64]
        w_down_chunk_0 = w_down_0[o0_3 : o0_3 + 32, d0_0 : d0_0 + 64]
        _t57 = torch.matmul(mlp_chunk_bf16_0, w_down_chunk_0)
        down_next_0 = torch.add(down_prev_0, _t57)
        down_proj_tile_6 [0: 0 + down_next_0.shape[0], d0_0: d0_0 + down_next_0.shape[1]] = down_next_0
    return down_proj_tile_6

if __name__ == '__main__':
    down_proj_tile_0 = torch.zeros([4, 5120], dtype=torch.float32, device='cuda')
    down_proj_tile_0_torch=torch.zeros_like(down_proj_tile_0)
    mlp_chunk_bf16_0 = torch.rand([4, 32], dtype = torch.bfloat16, device='cuda')
    w_down_0 = torch.rand([25600, 5120], dtype = torch.bfloat16, device='cuda')

    for b0_1 in range(0, 16, 4, ):
        for ob_5 in range(800, ):
            o0_3 = (ob_5 * 32)
            qwen3_decode_layer_incore_7[(20, 4, )](down_proj_tile_0, down_proj_tile_0.stride(0), down_proj_tile_0.stride(1), mlp_chunk_bf16_0, mlp_chunk_bf16_0.stride(0), mlp_chunk_bf16_0.stride(1), o0_3, w_down_0, w_down_0.stride(0), w_down_0.stride(1), )

            for dob_0_out in range(20):
                down_proj_tile_0_torch = qwen3_decode_layer_incore_7_torch(dob_0_out, down_proj_tile_0_torch, mlp_chunk_bf16_0, o0_3, w_down_0)

    print(torch.allclose(down_proj_tile_0, down_proj_tile_0_torch, atol=1e-3, rtol=1e-3))


    pass