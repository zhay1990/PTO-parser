import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_0(
    #pypto.language.Tensor[[16, 5120],pypto.language.BF16]
    hidden_states_0_ptr, hidden_states_0_stride_0, hidden_states_0_stride_1, 
    #pypto.language.Tensor[[16, 1],pypto.language.FP32]
    inv_rms_0_ptr, inv_rms_0_stride_0, inv_rms_0_stride_1, # Output ptr
):
    sq_sum_0 = tl.zeros([16, 1], dtype = tl.float32, )
    sq_sum_1 = sq_sum_0 * 0
    for kb_0 in range(20):
        k0_0 = (kb_0 * 256)
        _t0_offset = (0 + tl.arange(0, 16))[:, None] * hidden_states_0_stride_0 + (k0_0 + tl.arange(0, 256))[None, :] * hidden_states_0_stride_1
        _t0 = tl.load(hidden_states_0_ptr + _t0_offset)
        x_chunk_0 = _t0.to(tl.float32)
        _t1 = x_chunk_0 * x_chunk_0
        _t2 = tl.sum(_t1, axis = 1)[:, None]
        sq_sum_1 = sq_sum_1 + _t2
    _t3 = sq_sum_1 * 0.000195313
    _t4 = _t3 + 1e-06
    inv_rms_0 = tl.math.rsqrt(_t4)
    inv_rms_0_offset = (tl.arange(0, 16))[:, None] * inv_rms_0_stride_0 + (tl.arange(0, 1))[None, :] * inv_rms_0_stride_1
    tl.store(inv_rms_0_ptr + inv_rms_0_offset, inv_rms_0)

def qwen3_decode_layer_incore_0_torch(hidden_states_0):
    sq_sum_0 = torch.empty([16, 1], dtype = torch.float32, layout = torch.strided, device='cuda')
    sq_sum_1 = torch.mul(sq_sum_0, 0)
    for kb_0 in range(20):
        k0_0 = (kb_0 * 256)
        _t0 = hidden_states_0[0 : 0 + 16, k0_0 : k0_0 + 256]
        x_chunk_0 = _t0.to(torch.float32)
        _t1 = torch.mul(x_chunk_0, x_chunk_0)
        _t2 = torch.sum(_t1, dim = 1, keepdim = True)
        sq_sum_1 = torch.add(sq_sum_1, _t2)
    _t3 = torch.mul(sq_sum_1, 0.000195313)
    _t4 = torch.add(_t3, 1e-06)
    inv_rms_0 = torch.rsqrt(_t4)
    return inv_rms_0

if __name__ == '__main__':
    hidden_states_0 = torch.rand([16, 5120], dtype = torch.bfloat16, device='cuda')
    inv_rms_0 = torch.empty([16, 1], dtype = torch.float32, device='cuda')
    qwen3_decode_layer_incore_0[(1, )](hidden_states_0, hidden_states_0.stride(0), hidden_states_0.stride(1), inv_rms_0, inv_rms_0.stride(0), inv_rms_0.stride(1), )

    inv_rms = qwen3_decode_layer_incore_0_torch(hidden_states_0)

    # 没有太多的计算，精度要求可以高一点
    print(torch.allclose(inv_rms_0, inv_rms, atol=1e-3, rtol=1e-3))
    pass