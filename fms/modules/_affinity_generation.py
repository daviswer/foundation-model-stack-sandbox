import torch
from torch.autograd import Function
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_I': BLOCK_I, 'BLOCK_J': BLOCK_J}, num_stages=stages, num_warps=warps) \
    for BLOCK_I in [16, 32, 64, 128]\
    for BLOCK_J in [16, 32, 64, 128]\
    for stages in [1, 2, 3]\
    for warps in [2, 4, 8]\
]

# Optimal config tuned on A100
fwd_A100 = [triton.Config({'BLOCK_I': 16, 'BLOCK_J': 128}, num_stages=2, num_warps=4)]
bwd_A100 = [triton.Config({'BLOCK_I': 16, 'BLOCK_J': 32}, num_stages=2, num_warps=2)]
bwd_col_A100 = [triton.Config({'BLOCK_I': 64, 'BLOCK_J': 32}, num_stages=2, num_warps=2)]

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
@triton.autotune(
    configs=fwd_A100,
    key=['L', 'D'],
)
@triton.jit
def _aff_fwd_kernel(
    # Pointers to matrices
    k, src, dest, aff, 
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,             # b h l d
    str_src_b, str_src_h, str_src_li,               # b h l
    str_dest_b, str_dest_h, str_dest_lj,            # b h l
    str_aff_b, str_aff_h, str_aff_li, str_aff_lj,   # b h l l
    # Matrix dimensions                                         
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr, 
    BLOCK_D: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    err = 1e-12

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    aff_ptr = aff + pid_b * str_aff_b + pid_h * str_aff_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_i = tl.max_contiguous(tl.multiple_of(offs_i, BLOCK_I), BLOCK_I)

    src_vec = tl.load(src_ptr + offs_i * str_src_li, mask=(offs_i < L), other=0.0)
    src_mat = tl.sqrt_rn(src_vec + err)[:, None]

    prev_sum = tl.zeros((BLOCK_I,), dtype=tl.float32)

    for j_blocks in range(tl.cdiv(L, BLOCK_J)):
        offs_j = j_blocks * BLOCK_J + tl.arange(0, BLOCK_J)
        offs_j = tl.max_contiguous(tl.multiple_of(offs_j, BLOCK_J), BLOCK_J)

        dest_vec = tl.load(dest_ptr + offs_j * str_dest_lj, mask=(offs_j < L), other=0.0)
        dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

        affinity = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

        offs_d = tl.arange(0, BLOCK_D)
        offs_d = tl.max_contiguous(offs_d, BLOCK_D)
        k_mat = tl.load(k_ptr + offs_i[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
            mask=(offs_i[:, None] < L) & (offs_d[None, :] < D), other=0.0)
        kt_mat = tl.load(k_ptr + offs_j[:, None] * str_k_l + offs_d[None, :] * str_k_d, 
            mask=(offs_j[:, None] < L) & (offs_d[None, :] < D), other=0.0)

        # Use ieee to use fp32, otherwise the default would be tf32
        affinity += tl.dot(k_mat*src_mat, tl.trans(kt_mat*dest_mat), out_dtype=tl.float32)

        # .relu().pow(2/3), already in fp32
        affinity = tl.exp2(tl.log2(tl.maximum(affinity, err)) * 2.0 / 3.0)

        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        affinity = tl.log(1.0 - tl.clamp(affinity, 0.0, 1.0 - err)) 

        # .triu(1)
        affinity = tl.where((offs_i[:, None] < offs_j[None, :]), affinity, 0.0)

        # .cumsum(3)
        curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
        affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
        prev_sum += curr_sum
        
        # .masked_fill(mask.tril(-1), -1e12)
        affinity = tl.where((offs_i[:, None] > offs_j[None, :]), -1e12, affinity)

        tl.store(aff_ptr + offs_i[:, None] * str_aff_li + offs_j[None, :] * str_aff_lj, 
            affinity, mask=(offs_i[:, None] < L) & (offs_j[None, :] < L))

def _affinity_fwd(k, src, dest):
    '''
    Inputs:
    k:    b h l d
    src:  b h l
    dest: b h l

    Outputs:
    aff:  b h l l
    '''    
    b,h,l,d = k.shape
    assert src.shape == (b,h,l) and dest.shape == (b,h,l)

    aff = torch.empty(b,h,l,l, dtype=torch.float32, device=k.device)

    grid = lambda META: (b, h, triton.cdiv(l, META['BLOCK_I']))

    _aff_fwd_kernel[grid](
        k, src, dest, aff,                                                  
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2), 
        aff.stride(0), aff.stride(1), aff.stride(2), aff.stride(3), 
        B=b, H=h, L=l, D=d, BLOCK_D=d,
    )

    return aff.transpose(-1, -2).to(k.dtype)

'''
#######################################
#     Backward Kernel & Interface     #
#######################################
'''
@triton.autotune(
    configs=bwd_A100,
    key=['L', 'D'],
)
@triton.jit
def _aff_bwd_kernel(
    # Pointers to matrices
    k, src, dest, daff, daff_cs, dk_i, dsrc,                                                                                        
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,                     # b h l d
    str_src_b, str_src_h, str_src_li,                       # b h l
    str_dest_b, str_dest_h, str_dest_lj,                    # b h l
    str_daff_b, str_daff_h, str_daff_li, str_daff_lj,       # b h l l
    str_dac_b, str_dac_h, str_dac_li, str_dac_lj,           # b h l l
    str_dki_b, str_dki_h, str_dki_l, str_dki_d,             # b h l d
    str_dsrc_b, str_dsrc_h, str_dsrc_li,                    # b h l
    # Matrix dimensions                                         
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr, 
    BLOCK_D: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)
    err = 1e-12

    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    daff_ptr = daff + pid_b * str_daff_b + pid_h * str_daff_h
    dac_ptr = daff_cs + pid_b * str_dac_b + pid_h * str_dac_h
    dki_ptr = dk_i + pid_b * str_dki_b + pid_h * str_dki_h
    dsrc_ptr = dsrc + pid_b * str_dsrc_b + pid_h * str_dsrc_h

    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_i = tl.max_contiguous(tl.multiple_of(offs_i, BLOCK_I), BLOCK_I)
    offs_d_1 = tl.arange(0, BLOCK_D)

    src_vec = tl.load(src_ptr + offs_i * str_src_li, mask=(offs_i < L), other=0.0)
    src_mat = tl.sqrt_rn(src_vec + err)[:, None]

    suff_sum = tl.zeros((BLOCK_I,), dtype=tl.float32)
    dsrc_acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
    dk_i_acc_1 = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)

    for j_blocks in range(tl.cdiv(L, BLOCK_J) - 1, -1, -1):
        offs_j = j_blocks * BLOCK_J + tl.arange(0, BLOCK_J)
        offs_j = tl.max_contiguous(tl.multiple_of(offs_j, BLOCK_J), BLOCK_J)

        dest_vec = tl.load(dest_ptr + offs_j * str_dest_lj, mask=(offs_j < L), other=0.0)
        dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

        affinity_1 = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

        k_mat = tl.load(k_ptr + offs_i[:, None] * str_k_l + offs_d_1[None, :] * str_k_d, 
            mask=(offs_i[:, None] < L) & (offs_d_1[None, :] < D), other=0.0)
        kt_mat = tl.load(k_ptr + offs_j[:, None] * str_k_l + offs_d_1[None, :] * str_k_d, 
            mask=(offs_j[:, None] < L) & (offs_d_1[None, :] < D), other=0.0)
        affinity_1 += tl.dot(k_mat*src_mat, tl.trans(kt_mat*dest_mat), out_dtype=tl.float32)

        affinity_2 = tl.maximum(affinity_1, 0.0)
        affinity_3 = tl.exp2(tl.log2(affinity_2 + err) * 2.0 / 3.0)

        daffinity = tl.load(daff_ptr + offs_j[:, None] * str_daff_li + offs_i[None, :] * str_daff_lj, 
            mask=(offs_j[:, None] < L) & (offs_i[None, :] < L), other=0.0).cast(tl.float32)
        daffinity = tl.trans(daffinity)

        daffinity = tl.where((offs_i[:, None] > offs_j[None, :]), 0.0, daffinity)

        # Correct reverse cumsum: first compute local cumsum, then add suffix from previous blocks
        daffinity_cs = tl.cumsum(daffinity, axis=1, reverse=True)
        daffinity_cs += suff_sum[:, None]
        # Update suffix sum for next block (this should happen AFTER using the current suff_sum)
        suff_sum += tl.sum(daffinity, axis=1)
        
        daffinity = tl.where((offs_i[:, None] < offs_j[None, :]), daffinity_cs, 0.0)

        # Use numerically stable log gradient: d/dx log1p(-x) = -1/(1-x)
        # But clamp the denominator to avoid division by very small numbers
        daffinity = -daffinity / (1.0 - tl.clamp(affinity_3, 0.0, 1.0 - 1e-6))
        daffinity = daffinity * (2.0/3.0) * tl.exp2(tl.log2(affinity_2 + err) * (-1.0/3.0))
        daffinity = tl.where(affinity_1 > 0, daffinity, 0.0)

        tl.store(dac_ptr + offs_i[:, None] * str_dac_li + offs_j[None, :] * str_dac_lj, 
            daffinity, mask=(offs_i[:, None] < L) & (offs_j[None, :] < L))

        kt_mat = tl.load(k_ptr + offs_j[:, None] * str_k_l + offs_d_1[None, :] * str_k_d, 
            mask=(offs_j[:, None] < L) & (offs_d_1[None, :] < D), other=0.0)
        dk_i_acc_1 += tl.dot(daffinity, kt_mat*dest_mat, out_dtype=tl.float32)

        dsrc_acc += tl.sum(daffinity * affinity_1.cast(tl.bfloat16), axis=1) 
           
    tl.store(dsrc_ptr + offs_i * str_dsrc_li, tl.div_rn(dsrc_acc, 2.0*(src_vec+err)), mask=(offs_i < L))

    tl.store(dki_ptr + offs_i[:, None] * str_dki_l + offs_d_1[None, :] * str_dki_d, 
        dk_i_acc_1.cast(tl.bfloat16) * src_mat, mask=(offs_i[:, None] < L) & (offs_d_1[None, :] < D))

@triton.autotune(
    configs=bwd_col_A100,
    key=['L', 'D'],
)
@triton.jit
def _aff_bwd_col_kernel(
    # Pointers to matrices
    k, src, dest, daff_cs, dk_j, ddest,
    # Strides
    str_k_b, str_k_h, str_k_l, str_k_d,                     # b h l d
    str_src_b, str_src_h, str_src_li,                       # b h l
    str_dest_b, str_dest_h, str_dest_lj,                    # b h l
    str_dac_b, str_dac_h, str_dac_li, str_dac_lj,           # b h l l
    str_dkj_b, str_dkj_h, str_dkj_l, str_dkj_d,             # b h l d
    str_ddest_b, str_ddest_h, str_ddest_lj,                 # b h l
    # Matrix dimensions
    B: tl.constexpr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    # Meta-parameters
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_j = tl.program_id(2)
    err = 1e-12
    
    k_ptr = k + pid_b * str_k_b + pid_h * str_k_h
    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h
    dest_ptr = dest + pid_b * str_dest_b + pid_h * str_dest_h
    dac_ptr = daff_cs + pid_b * str_dac_b + pid_h * str_dac_h
    dkj_ptr = dk_j + pid_b * str_dkj_b + pid_h * str_dkj_h
    ddest_ptr = ddest + pid_b * str_ddest_b + pid_h * str_ddest_h

    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_d_1 = tl.arange(0, BLOCK_D)

    dest_vec = tl.load(dest_ptr + offs_j * str_dest_lj, mask=(offs_j < L), other=0.0)
    dest_mat = tl.sqrt_rn(dest_vec + err)[:, None]

    ddest_acc = tl.zeros((BLOCK_J,), dtype=tl.float32)
    dk_j_acc_1 = tl.zeros((BLOCK_J, BLOCK_D), dtype=tl.float32)


    for i_offset in range(0, L, BLOCK_I):
        offs_i = i_offset + tl.arange(0, BLOCK_I)

        src_vec = tl.load(src_ptr + offs_i * str_src_li, mask=(offs_i < L), other=0.0)
        src_mat = tl.sqrt_rn(src_vec + err)[:, None]

        daffinity = tl.load(dac_ptr + offs_i[:, None] * str_dac_li + offs_j[None, :] * str_dac_lj,
            mask=(offs_i[:, None] < L) & (offs_j[None, :] < L), other=0.0)

        affinity_1 = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)

        k_mat = tl.load(k_ptr + offs_i[:, None] * str_k_l + offs_d_1[None, :] * str_k_d, 
            mask=(offs_i[:, None] < L) & (offs_d_1[None, :] < D), other=0.0)
        kt_mat = tl.load(k_ptr + offs_j[:, None] * str_k_l + offs_d_1[None, :] * str_k_d, 
            mask=(offs_j[:, None] < L) & (offs_d_1[None, :] < D), other=0.0)
        k_src = k_mat * src_mat
        affinity_1 += tl.dot(k_src, tl.trans(kt_mat*dest_mat), out_dtype=tl.float32)
        dk_j_acc_1 += tl.dot(tl.trans(daffinity), k_src, out_dtype=tl.float32)

        ddest_acc += tl.sum(daffinity * affinity_1.cast(tl.bfloat16), axis=0) 

    tl.store(ddest_ptr + offs_j * str_ddest_lj, tl.div_rn(ddest_acc, 2.0 * (dest_vec + err)), mask=(offs_j < L))

    tl.store(dkj_ptr + offs_j[:, None] * str_dkj_l + offs_d_1[None, :] * str_dkj_d, 
        dk_j_acc_1.cast(tl.bfloat16) * dest_mat, mask=(offs_j[:, None] < L) & (offs_d_1[None, :] < D))
    
def _affinity_bwd(k, src, dest, daff):
    '''
    Inputs:
    k:     b h l d
    src:   b h l
    dest:  b h l
    daff:  b h l l

    Outputs:
    dk:    b h l d
    dsrc:  b h l
    ddest: b h l
    '''    
    b,h,l,d = k.shape
    assert src.shape == (b,h,l) and dest.shape == (b,h,l) and daff.shape == (b,h,l,l)

    daff_cs = torch.empty(b,h,l,l, dtype=torch.float32, device=daff.device)
    dk_i = torch.empty_like(k, dtype=torch.bfloat16)
    dk_j = torch.empty_like(k, dtype=torch.bfloat16)
    dsrc = torch.empty_like(src, dtype=torch.bfloat16)
    ddest = torch.empty_like(dest, dtype=torch.bfloat16)

    grid_i = lambda META: (b, h, triton.cdiv(l, META['BLOCK_I']))
    _aff_bwd_kernel[grid_i](
        k, src, dest, daff, daff_cs, dk_i, dsrc, 
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
        src.stride(0), src.stride(1), src.stride(2),  
        dest.stride(0), dest.stride(1), dest.stride(2),      
        daff.stride(0), daff.stride(1), daff.stride(2), daff.stride(3),     
        daff_cs.stride(0), daff_cs.stride(1), daff_cs.stride(2), daff_cs.stride(3),       
        dk_i.stride(0), dk_i.stride(1), dk_i.stride(2), dk_i.stride(3), 
        dsrc.stride(0), dsrc.stride(1), dsrc.stride(2),  
        B=b, H=h, L=l, D=d, BLOCK_D=d
    )

    grid_j = lambda META: (b, h, triton.cdiv(l, META['BLOCK_J']))
    _aff_bwd_col_kernel[grid_j](
        k, src, dest, daff_cs, dk_j, ddest,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        src.stride(0), src.stride(1), src.stride(2), 
        dest.stride(0), dest.stride(1), dest.stride(2),
        daff_cs.stride(0), daff_cs.stride(1), daff_cs.stride(2), daff_cs.stride(3), 
        dk_j.stride(0), dk_j.stride(1), dk_j.stride(2), dk_j.stride(3),
        ddest.stride(0), ddest.stride(1), ddest.stride(2), 
        B=b, H=h, L=l, D=d, BLOCK_D=d
    )

    dk = dk_i + dk_j

    return dk.to(k.dtype), dsrc.to(src.dtype), ddest.to(dest.dtype)

class AffinityScores(Function):
    @staticmethod
    def forward(k, src, dest):
        aff = _affinity_fwd(k, src, dest)
        return aff
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        k, src, dest = inputs
        # affinity = outputs
        ctx.save_for_backward(k, src, dest)
    
    @staticmethod
    def backward(ctx, daff):
        k, src, dest = ctx.saved_tensors
        if daff is None:
            return (None, None, None)
        daff = daff.contiguous()
        dk_, dsrc_, ddest_ = _affinity_bwd(k, src, dest, daff)

        dk    = dk_    if ctx.needs_input_grad[0] else None
        dsrc  = dsrc_  if ctx.needs_input_grad[1] else None
        ddest = ddest_ if ctx.needs_input_grad[2] else None
        return dk, dsrc, ddest
    
_gen_affinity_scores = AffinityScores.apply

'''
#################################
#     Ground Truth Autograd     #
#################################
'''
def _gen_affinity_scores_torch(k, src, dest):
    kkt = torch.einsum('bnqh, bnkh -> bnqk', k, k).relu().pow(2/3).float()
    affinity = kkt * src.pow(1/3).unsqueeze(-1) * dest.pow(1/3).unsqueeze(-2)
    affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = affinity.triu(1).cumsum(3).to(dtype=k.dtype)
    return torch.transpose(affinity.masked_fill(torch.ones_like(affinity, dtype=torch.bool).tril(-1), -1e12), -1, -2).contiguous()

def _gen_affinity_scores_torch_fused(k, src, dest):
    affinity = torch.einsum('bnqh, bnkh -> bnqk', k*src.sqrt().unsqueeze(-1), k*dest.sqrt().unsqueeze(-1)).relu().float().pow(2/3)
    affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
    affinity = affinity.triu(1).cumsum(3).to(dtype=k.dtype)
    return torch.transpose(affinity.masked_fill(torch.ones_like(affinity, dtype=torch.bool).tril(-1), -1e12), -1, -2).contiguous()

def get_optimal_configs():
    print("Optimal auto-tuned configs")
    print("_aff_fwd_kernel")
    print(_aff_fwd_kernel.best_config)
    print("_aff_bwd_kernel")
    print(_aff_bwd_kernel.best_config)
    print("_aff_bwd_col_kernel")
    print(_aff_bwd_col_kernel.best_config)