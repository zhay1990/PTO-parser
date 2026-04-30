import triton
import triton.language as tl
import torch

@triton.jit
def qwen3_decode_layer_incore_4(
    #pypto.language.Tensor[[1, 8192],pypto.language.FP32]
    attn_row_4_ptr, attn_row_4_stride_0, attn_row_4_stride_1, 
    b_0,
    #pypto.language.Tensor[[1, 64],pypto.language.FP32]
    cos_hi_0_ptr, cos_hi_0_stride_0, cos_hi_0_stride_1, 
    #pypto.language.Tensor[[1, 64],pypto.language.FP32]
    cos_lo_0_ptr, cos_lo_0_stride_0, cos_lo_0_stride_1, 
    ctx_blocks_0,
    ctx_len_0,
    #pypto.language.Tensor[[524288, 128],pypto.language.BF16]
    k_cache_4_ptr, k_cache_4_stride_0, k_cache_4_stride_1, 
    #pypto.language.Tensor[[16, 8192],pypto.language.BF16]
    q_proj_2_ptr, q_proj_2_stride_0, q_proj_2_stride_1, 
    #pypto.language.Tensor[[1, 64],pypto.language.FP32]
    sin_hi_0_ptr, sin_hi_0_stride_0, sin_hi_0_stride_1, 
    #pypto.language.Tensor[[1, 64],pypto.language.FP32]
    sin_lo_0_ptr, sin_lo_0_stride_0, sin_lo_0_stride_1, 
    #pypto.language.Tensor[[524288, 128],pypto.language.BF16]
    v_cache_4_ptr, v_cache_4_stride_0, v_cache_4_stride_1, 
    # Scatch pad memory tensor<[1, 128], FP32>
    q_row_0_scratchPad_ptr, q_row_0_scratchPad_stride_0, q_row_0_scratchPad_stride_1, # Output ptr
):
    h_0_out = tl.program_id(axis = 0)
    h_0_in = tl.program_id(axis = 1)
    # 实际尺寸和原始尺寸一致, 不需要mask
    sin_hi_0_offset = (tl.arange(0, 1))[:, None] * sin_hi_0_stride_0 + (tl.arange(0, 64))[None, :] * sin_hi_0_stride_1
    sin_hi_0 = tl.load(sin_hi_0_ptr + sin_hi_0_offset)
    # 实际尺寸和原始尺寸一致, 不需要mask
    cos_hi_0_offset = (tl.arange(0, 1))[:, None] * cos_hi_0_stride_0 + (tl.arange(0, 64))[None, :] * cos_hi_0_stride_1
    cos_hi_0 = tl.load(cos_hi_0_ptr + cos_hi_0_offset)
    # 实际尺寸和原始尺寸一致, 不需要mask
    sin_lo_0_offset = (tl.arange(0, 1))[:, None] * sin_lo_0_stride_0 + (tl.arange(0, 64))[None, :] * sin_lo_0_stride_1
    sin_lo_0 = tl.load(sin_lo_0_ptr + sin_lo_0_offset)
    # 实际尺寸和原始尺寸一致, 不需要mask
    cos_lo_0_offset = (tl.arange(0, 1))[:, None] * cos_lo_0_stride_0 + (tl.arange(0, 64))[None, :] * cos_lo_0_stride_1
    cos_lo_0 = tl.load(cos_lo_0_ptr + cos_lo_0_offset)
    kvh_3 = ((0 + (((h_0_out * 8) + h_0_in) * 1)) // 8)
    q_col_0 = ((0 + (((h_0_out * 8) + h_0_in) * 1)) * 128)
    # 因为输入的PTO源码没有边界检查, 所以没有为这个tl.load生成Mask
    _t23_offset = (b_0 + tl.arange(0, 1))[:, None] * q_proj_2_stride_0 + (q_col_0 + tl.arange(0, 128))[None, :] * q_proj_2_stride_1
    _t23 = tl.load(q_proj_2_ptr + _t23_offset)
    q_row_0 = _t23.to(tl.float32)
    # Slice conversion: store offset
    q_row_0_store_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
    tl.store(q_row_0_scratchPad_ptr + q_row_0_store_offset, q_row_0)
    # Slice conversion: load offset
    q_row_0_load_offset = (0 + tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (0 + tl.arange(0, 64))[None, :] * q_row_0_scratchPad_stride_1
    q_row_0_load_mask = ((tl.arange(0, 1))[:, None] < 1) &  ((tl.arange(0, 64))[None, :] < 64)
    q_lo_0 = tl.load(q_row_0_scratchPad_ptr + q_row_0_load_offset, mask=q_row_0_load_mask, other=0.0)
    # Slice conversion: store offset
    q_row_0_store_offset_ = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
    tl.store(q_row_0_scratchPad_ptr + q_row_0_store_offset_, q_row_0)
    # Slice conversion: load offset
    q_row_0_load_offset_ = (0 + tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (64 + tl.arange(0, 64))[None, :] * q_row_0_scratchPad_stride_1
    q_row_0_load_mask_ = ((tl.arange(0, 1))[:, None] < 1) &  ((tl.arange(0, 64))[None, :] < 64)
    q_hi_0 = tl.load(q_row_0_scratchPad_ptr + q_row_0_load_offset_, mask=q_row_0_load_mask_, other=0.0)
    q_rot_0 = tl.zeros([1, 128], dtype = tl.float32, )
    _t24 = q_lo_0 * cos_lo_0
    _t25 = q_hi_0 * sin_lo_0
    _t26 = _t24 - _t25
    # Assemble conversion: store offset
    q_rot_0_store_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
    tl.store(q_row_0_scratchPad_ptr + q_rot_0_store_offset, q_rot_0)
    # Assemble conversion: store offset #2
    _t26_store_offset = (0 + tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (0 + tl.arange(0, 64))[None, :] * q_row_0_scratchPad_stride_1
    _t26_store_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 64))[None, :] < 64)
    tl.store(q_row_0_scratchPad_ptr + _t26_store_offset, _t26, mask=_t26_store_mask)
    # Assemble conversion: load offset
    q_rot_1_load_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
    q_rot_1_load_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 128))[None, :] < 128)
    q_rot_1 = tl.load(q_row_0_scratchPad_ptr + q_rot_1_load_offset, mask=q_rot_1_load_mask, other=0.0)
    _t27 = q_hi_0 * cos_hi_0
    _t28 = q_lo_0 * sin_hi_0
    _t29 = _t27 + _t28
    # Assemble conversion: store offset
    q_rot_1_store_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
    tl.store(q_row_0_scratchPad_ptr + q_rot_1_store_offset, q_rot_1)
    # Assemble conversion: store offset #2
    _t29_store_offset = (0 + tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (64 + tl.arange(0, 64))[None, :] * q_row_0_scratchPad_stride_1
    _t29_store_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 64))[None, :] < 64)
    tl.store(q_row_0_scratchPad_ptr + _t29_store_offset, _t29, mask=_t29_store_mask)
    # Assemble conversion: load offset
    q_rot_2_load_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
    q_rot_2_load_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 128))[None, :] < 128)
    q_rot_2 = tl.load(q_row_0_scratchPad_ptr + q_rot_2_load_offset, mask=q_rot_2_load_mask, other=0.0)
    q_rot_bf16_0 = q_rot_2.to(tl.bfloat16)
    oi_0 = tl.zeros([1, 128], dtype = tl.float32, )
    li_0 = tl.zeros([1, 1], dtype = tl.float32, )
    mi_0 = tl.zeros([1, 1], dtype = tl.float32, )
    oi_1 = oi_0 * 0
    li_1 = li_0 * 0
    mi_1 = mi_0 * 0
    for sb_0 in range(ctx_blocks_0):
        s0_0 = (sb_0 * 120)
        valid_len = tl.minimum(120, (ctx_len_0.to(tl.int32) - s0_0))
        cache_row0_0 = ((((b_0 * 8) * 4096) + (kvh_3 * 4096)) + s0_0)
        # k_tile_0的shape从[120, 128]拓展成了[128, 128], 多出来的数据填0
        k_tile_0_offset = (cache_row0_0 + tl.arange(0, 128))[:, None] * k_cache_4_stride_0 + (0 + tl.arange(0, 128))[None, :] * k_cache_4_stride_1
        # 基于有效数据范围[valid_len, 128]生成Mask
        k_tile_0_mask = ((tl.arange(0, 128))[:, None] < valid_len) & ((tl.arange(0, 128))[None, :] < 128)
        k_tile_0 = tl.load(k_cache_4_ptr + k_tile_0_offset, mask=k_tile_0_mask, other=0.0)
        # v_tile_0的shape从[120, 128]拓展成了[128, 128], 多出来的数据填0
        v_tile_0_offset = (cache_row0_0 + tl.arange(0, 128))[:, None] * v_cache_4_stride_0 + (0 + tl.arange(0, 128))[None, :] * v_cache_4_stride_1
        # 基于有效数据范围[valid_len, 128]生成Mask
        v_tile_0_mask = ((tl.arange(0, 128))[:, None] < valid_len) & ((tl.arange(0, 128))[None, :] < 128)
        v_tile_0 = tl.load(v_cache_4_ptr + v_tile_0_offset, mask=v_tile_0_mask, other=0.0)
        _t30 = tl.dot(q_rot_bf16_0, tl.trans(k_tile_0))
        scores_0 = _t30 * 0.0883883
        # Slice conversion: store offset
        scores_0_store_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
        tl.store(q_row_0_scratchPad_ptr + scores_0_store_offset, scores_0)
        # Slice conversion: load offset
        scores_0_load_offset = (0 + tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (0 + tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
        scores_0_load_mask = ((tl.arange(0, 1))[:, None] < 1) &  ((tl.arange(0, 128))[None, :] < valid_len)
        scores_valid_0 = tl.load(q_row_0_scratchPad_ptr + scores_0_load_offset, mask=scores_0_load_mask, other=0.0)
        # 将无效数据替换为-inf
        scores_valid_0_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 128))[None, :] < valid_len)
        scores_valid_0 = tl.where(scores_valid_0_mask, scores_valid_0, float('-inf'))
        _t31 = tl.max(scores_valid_0, axis = 1)[:, None]
        # 将无效数据替换回0.0, 适配除max以外的其他计算
        scores_valid_0 = tl.where(scores_valid_0_mask, scores_valid_0, 0.0)
        cur_mi_0 = _t31.to(tl.float32)
        _t32 = scores_valid_0 - cur_mi_0
        exp_scores_0 = tl.exp(_t32)
        _t33 = tl.sum(exp_scores_0, axis = 1)[:, None]
        cur_li_0 = _t33.to(tl.float32)
        exp_pad_0 = tl.zeros([1, 128], dtype = tl.float32, )
        exp_pad_1 = exp_pad_0 * 0
        # Assemble conversion: store offset
        exp_pad_1_store_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
        tl.store(q_row_0_scratchPad_ptr + exp_pad_1_store_offset, exp_pad_1)
        # Assemble conversion: store offset #2
        exp_scores_0_store_offset = (0 + tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (0 + tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
        exp_scores_0_store_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 128))[None, :] < valid_len)
        tl.store(q_row_0_scratchPad_ptr + exp_scores_0_store_offset, exp_scores_0, mask=exp_scores_0_store_mask)
        # Assemble conversion: load offset
        exp_pad_2_load_offset = (tl.arange(0, 1))[:, None] * q_row_0_scratchPad_stride_0 + (tl.arange(0, 128))[None, :] * q_row_0_scratchPad_stride_1
        exp_pad_2_load_mask = ((tl.arange(0, 1))[:, None] < 1) & ((tl.arange(0, 128))[None, :] < 120)
        exp_pad_2 = tl.load(q_row_0_scratchPad_ptr + exp_pad_2_load_offset, mask=exp_pad_2_load_mask, other=0.0)
        _t34 = exp_pad_2.to(tl.bfloat16)
        oi_tmp_0 = tl.dot(_t34, v_tile_0)
        if (sb_0 == 0):
            oi_4 = oi_tmp_0
            li_4 = cur_li_0
            mi_4 = cur_mi_0
            li_1 = li_4
            mi_1 = mi_4
            oi_1 = oi_4
        else:
            mi_new_0 = tl.maximum(mi_1, cur_mi_0)
            _t35 = mi_1 - mi_new_0
            alpha_0 = tl.exp(_t35)
            _t36 = cur_mi_0 - mi_new_0
            beta_0 = tl.exp(_t36)
            _t37 = alpha_0 * li_1
            _t38 = beta_0 * cur_li_0
            li_5 = _t37 + _t38
            _t39 = oi_1 * alpha_0
            _t40 = oi_tmp_0 * beta_0
            oi_5 = _t39 + _t40
            mi_5 = mi_new_0
            li_1 = li_5
            mi_1 = mi_5
            oi_1 = oi_5

    ctx_0 = oi_1 / li_1
    attn_row_4_offset = (0 + tl.arange(0, 1))[:, None] * attn_row_4_stride_0 + (q_col_0 + tl.arange(0, 128))[None, :] * attn_row_4_stride_1
    tl.store(attn_row_4_ptr + attn_row_4_offset, ctx_0)

def qwen3_decode_layer_incore_4_torch(attn_row_4, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_4, q_proj_2, sin_hi_0, sin_lo_0, v_cache_4):
    for h_0_in in range(8):
        kvh_3 = ((0 + (((h_0_out * 8) + h_0_in) * 1)) // 8)
        q_col_0 = ((0 + (((h_0_out * 8) + h_0_in) * 1)) * 128)
        _t23 = q_proj_2[b_0 : b_0 + 1, q_col_0 : q_col_0 + 128]
        q_row_0 = _t23.to(torch.float32)
        q_lo_0 = q_row_0[0 : 0 + 1, 0 : 0 + 64]
        q_hi_0 = q_row_0[0 : 0 + 1, 64 : 64 + 64]
        q_rot_0 = torch.empty([1, 128], dtype = torch.float32, layout = torch.strided, device='cuda')
        _t24 = q_lo_0 * cos_lo_0
        _t25 = q_hi_0 * sin_lo_0
        _t26 = torch.sub(_t24, _t25)
        q_rot_1 = q_rot_0
        q_rot_1 [0: 0 + _t26.shape[0], 0: 0 + _t26.shape[1]] = _t26
        _t27 = q_hi_0 * cos_hi_0
        _t28 = q_lo_0 * sin_hi_0
        _t29 = torch.add(_t27, _t28)
        q_rot_2 = q_rot_1
        q_rot_2 [0: 0 + _t29.shape[0], 64: 64 + _t29.shape[1]] = _t29
        q_rot_bf16_0 = q_rot_2.to(torch.bfloat16)
        oi_0 = torch.empty([1, 128], dtype = torch.float32, layout = torch.strided, device='cuda')
        li_0 = torch.empty([1, 1], dtype = torch.float32, layout = torch.strided, device='cuda')
        mi_0 = torch.empty([1, 1], dtype = torch.float32, layout = torch.strided, device='cuda')
        oi_1 = torch.mul(oi_0, 0)
        li_1 = torch.mul(li_0, 0)
        mi_1 = torch.mul(mi_0, 0)
        for sb_0 in range(ctx_blocks_0):
            s0_0 = (sb_0 * 120)
            valid_len = min(120, (int(ctx_len_0) - s0_0))
            cache_row0_0 = ((((b_0 * 8) * 4096) + (kvh_3 * 4096)) + s0_0)
            k_tile_0 = torch.zeros([120, 128], dtype=torch.bfloat16, device='cuda')
            k_tile_0[0: valid_len, 0: 128] = k_cache_4[cache_row0_0 : cache_row0_0 + valid_len, 0 : 0 + 128]
            v_tile_0 = torch.zeros([120, 128], dtype=torch.bfloat16, device='cuda')
            v_tile_0[0: valid_len, 0: 128] = v_cache_4[cache_row0_0 : cache_row0_0 + valid_len, 0 : 0 + 128]
            _t30 = torch.matmul(q_rot_bf16_0, k_tile_0.mT)
            scores_0 = torch.mul(_t30, 0.0883883)
            scores_valid_0 = scores_0[0 : 0 + 1, 0 : 0 + valid_len]
            _t31 = torch.max(scores_valid_0, dim = 1, keepdim = True)[0]
            cur_mi_0 = _t31.to(torch.float32)
            _t32 = scores_valid_0 - cur_mi_0
            exp_scores_0 = torch.exp(_t32)
            _t33 = torch.sum(exp_scores_0, dim = 1, keepdim = True)
            cur_li_0 = _t33.to(torch.float32)
            exp_pad_0 = torch.empty([1, 120], dtype = torch.float32, layout = torch.strided, device='cuda')
            exp_pad_1 = torch.mul(exp_pad_0, 0)
            exp_pad_2 = exp_pad_1
            exp_pad_2 [0: 0 + exp_scores_0.shape[0], 0: 0 + exp_scores_0.shape[1]] = exp_scores_0
            _t34 = exp_pad_2.to(torch.bfloat16)
            oi_tmp_0 = torch.matmul(_t34, v_tile_0)
            if (sb_0 == 0):
                oi_4 = oi_tmp_0
                li_4 = cur_li_0
                mi_4 = cur_mi_0
                li_1 = li_4
                mi_1 = mi_4
                oi_1 = oi_4
            else:
                mi_new_0 = torch.maximum(mi_1, cur_mi_0)
                _t35 = torch.sub(mi_1, mi_new_0)
                alpha_0 = torch.exp(_t35)
                _t36 = torch.sub(cur_mi_0, mi_new_0)
                beta_0 = torch.exp(_t36)
                _t37 = torch.mul(alpha_0, li_1)
                _t38 = torch.mul(beta_0, cur_li_0)
                li_5 = torch.add(_t37, _t38)
                _t39 = oi_1 * alpha_0
                _t40 = oi_tmp_0 * beta_0
                oi_5 = torch.add(_t39, _t40)
                mi_5 = mi_new_0
                li_1 = li_5
                mi_1 = mi_5
                oi_1 = oi_5
        ctx_0 = oi_1 / li_1
        attn_row_4 [0: 0 + ctx_0.shape[0], q_col_0: q_col_0 + ctx_0.shape[1]] = ctx_0
    return attn_row_4

if __name__ == '__main__':
    attn_row_1 = torch.empty([1, 8192], dtype = torch.float32, device='cuda')
    attn_row_torch = torch.empty([1, 8192], dtype = torch.float32, device='cuda')
    rope_cos_0 = torch.rand([4096, 128], dtype = torch.float32, device='cuda')
    rope_sin_0 = torch.rand([4096, 128], dtype = torch.float32, device='cuda')
    k_cache_0 = torch.rand([524288, 128], dtype = torch.bfloat16, device='cuda')
    v_cache_0 = torch.rand([524288, 128], dtype = torch.bfloat16, device='cuda')
    q_proj_0 = torch.rand([16, 8192], dtype = torch.bfloat16, layout = torch.strided, device='cuda')
    seq_lens_0 = torch.randint(1, 4096, [16], dtype = torch.int32)

    for b_0 in range(16, ):
        ctx_len_0 = seq_lens_0[b_0]
        pos_0 = (int(ctx_len_0) - 1)
        ctx_blocks_0 = (((int(ctx_len_0) + 120) - 1) // 120)
        cos_row_0 = rope_cos_0[pos_0 : pos_0 + 1, 0 : 0 + 128]
        sin_row_0 = rope_sin_0[pos_0 : pos_0 + 1, 0 : 0 + 128]

        cos_lo_0 = cos_row_0[0 : 0 + 1, 0 : 0 + 64]
        cos_hi_0 = cos_row_0[0 : 0 + 1, 64 : 64 + 64]
        sin_lo_0 = sin_row_0[0 : 0 + 1, 0 : 0 + 64]
        sin_hi_0 = sin_row_0[0 : 0 + 1, 64 : 64 + 64]

        __scratchPadMem_0 = torch.empty([1, 128], dtype = torch.float32, device = 'cuda')
        qwen3_decode_layer_incore_4[(8, 8, )](attn_row_1, attn_row_1.stride(0), attn_row_1.stride(1), int(b_0), cos_hi_0, cos_hi_0.stride(0), cos_hi_0.stride(1), cos_lo_0, cos_lo_0.stride(0), cos_lo_0.stride(1), int(ctx_blocks_0), int(ctx_len_0), k_cache_0, k_cache_0.stride(0), k_cache_0.stride(1), q_proj_0, q_proj_0.stride(0), q_proj_0.stride(1), sin_hi_0, sin_hi_0.stride(0), sin_hi_0.stride(1), sin_lo_0, sin_lo_0.stride(0), sin_lo_0.stride(1), v_cache_0, v_cache_0.stride(0), v_cache_0.stride(1), __scratchPadMem_0, __scratchPadMem_0.stride(0), __scratchPadMem_0.stride(1), num_stages=1)

        for h_0_out in range(8):
            attn_row_torch = qwen3_decode_layer_incore_4_torch(attn_row_torch, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_0, q_proj_0, sin_hi_0, sin_lo_0, v_cache_0)

    print(torch.allclose(attn_row_1, attn_row_torch, atol=1e-2, rtol=1e-2))

    abs_diff = torch.abs(attn_row_1 - attn_row_torch)
    max_diff = abs_diff.max().item()
    print(f"🔥 最大绝对误差 (Max Abs Diff): {max_diff:.6f}")

    pass