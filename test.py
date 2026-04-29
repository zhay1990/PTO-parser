import torch
import triton
import triton.language as tl

@triton.jit
def dynamic_slice_embed_kernel(
    in_ptr, 
    out_ptr,
    start_i,               # 【新增】：从外部传入截取起始列 i
    BLOCK_M: tl.constexpr, # 16
    BLOCK_K: tl.constexpr, # 128 (原始长度)
    BLOCK_N: tl.constexpr  # 64  (截取长度)
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    
    # 1. 加载原始张量 A [16, 128]
    a_ptrs = in_ptr + (offs_m[:, None] * BLOCK_K + offs_k[None, :])
    A = tl.load(a_ptrs).to(tl.float16)
    
    # ---------------------------------------------------------
    # 2. 【核心】：构造动态滑动的投影矩阵 T [128, 64]
    # ---------------------------------------------------------
    # 条件：T 的行索引 (offs_k) == 列索引 (offs_n) + 起始偏移量 (start_i)
    T_f32 = tl.where(offs_k[:, None] == (offs_n[None, :] + start_i), 1.0, 0.0)
    T = T_f32.to(tl.float16)
    
    # 3. 执行降维切片：[16, 128] @ [128, 64] -> [16, 64]
    C = tl.dot(A, T)
    
    # 4. 将截取后的结果写回显存
    c_ptrs = out_ptr + (offs_m[:, None] * BLOCK_N + offs_n[None, :])
    tl.store(c_ptrs, C.to(tl.float32))


# ==========================================
# Python 端验证测试
# ==========================================
def test_dynamic_slice():
    BLOCK_M = 16
    BLOCK_K = 128
    BLOCK_N = 64
    
    # 为了直观验证，我们让输入 A 的值等于它的列索引
    # [0, 1, 2, ..., 127]
    a_row = torch.arange(0, BLOCK_K, dtype=torch.float16, device='cuda')
    # 扩展成 16 行
    a = a_row.unsqueeze(0).expand(BLOCK_M, BLOCK_K).contiguous()
    
    # 测试不同的偏移量 i
    test_indices = [0, 32, 64]
    
    for i in test_indices:
        out_c = torch.empty((BLOCK_M, BLOCK_N), device='cuda', dtype=torch.float32)
        
        # 将 i 传进去
        dynamic_slice_embed_kernel[(1,)](a, out_c, i, BLOCK_M, BLOCK_K, BLOCK_N)
        
        print(f"\n=== 测试截取偏移量 i = {i} ===")
        print(f"预期提取结果: {i} 到 {i + 63}")
        
        # 取第 0 行，打印前 5 个和后 5 个元素验证
        res = out_c[0].cpu().tolist()
        print(f"实际输出结果: {res[:5]} ... {res[-5:]}")

if __name__ == "__main__":
    test_dynamic_slice()