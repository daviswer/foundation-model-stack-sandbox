import abc
import functools
import math
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Unpack,
)

import torch
import torch.distributed
from torch import Tensor, nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional as F

from fms import distributed
from fms.distributed.tensorparallel import (
    copy_to_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
)
from fms.modules.linear import (
    LinearModuleShardingInfo,
    get_all_linear_type_to_sharding_maps,
    get_linear,
    get_linear_type,
)
from fms.modules.positions import PositionEncoder
from fms.modules.tp import TPModule

from torch.autograd import Function



import torch
import triton
import triton.language as tl

import numpy as np
import inspect
import time

configs = [
    triton.Config({'BLOCK_D': BLOCK_D}, num_stages=stages, num_warps=warps) \
    for BLOCK_D in [32, 64, 128]\
    for stages in [2, 3, 4]\
    for warps in [2, 4, 8]\
]

'''
######################################
#     Forward Kernel & Interface     #
######################################
'''
# @triton.autotune(
#     configs=configs,
#     key=['r', 'n_', '_n', 'd'],
# )
@triton.jit
def _universal_attention_fwd_kernel(
    # Pointers to matrices
    kc, vc, xq, kt, src, dest, out, denom,                                
    # Matrix dimensions
    b, h, r, n_, _n, d, 
    # Strides
    str_kc_b, str_kc_h, str_kc_n_, str_kc_c_, str_kc_d,                     # b h n_ c_ d
    str_vc_b, str_vc_h, str_vc_n_, str_vc_c_, str_vc_d,                     # b h n_ c_ d
    str_xq_b, str_xq_h, str_xq_r, str_xq__n, str_xq__c, str_xq_d,           # b h r _n _c d
    str_kt_b, str_kt_h, str_kt_d, str_kt__n, str_kt__c,                     # b h d _n _c
    str_src_b, str_src_h, str_src_n_, str_src_c_,                           # b h n_ c_
    str_dest_b, str_dest_h, str_dest__n, str_dest__c,                       # b h _n _c
    str_out_b, str_out_h, str_out_r, str_out_l, str_out_d, str_out_n_,      # b h r l d n_
    str_denom_b, str_denom_h, str_denom_r, str_denom_l, str_denom_n_,       # b h r l n_
    # Meta-parameters
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,    # Block dims
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)                # n_

    offs_i = tl.arange(0, BLOCK_C)          # c_
    offs_j = tl.arange(0, BLOCK_R)          # _c
    offs_k = tl.arange(0, BLOCK_D)          # d

    kc_ptr = kc + pid_b * str_kc_b + pid_h * str_kc_h + pid_i * str_kc_n_
    vc_ptr = vc + pid_b * str_vc_b + pid_h * str_vc_h + pid_i * str_vc_n_

    src_ptr = src + pid_b * str_src_b + pid_h * str_src_h + pid_i * str_src_n_ 
    src_mat = tl.load(src_ptr + offs_i * str_src_c_, mask=offs_i < BLOCK_C, other=0.0)
    src_mat = tl.cast(src_mat, tl.float32)
    src_mat = tl.exp2(tl.log2(src_mat) / 3.0)

    # Pointers that depend on j
    xq_j = xq + pid_b * str_xq_b + pid_h * str_xq_h
    kt_j = kt + pid_b * str_kt_b + pid_h * str_kt_h 
    out_j = out + pid_b * str_out_b + pid_h * str_out_h + pid_i * str_out_n_
    dest_j = dest + pid_b * str_dest_b + pid_h * str_dest_h
    denom_j = denom + pid_b * str_denom_b + pid_h * str_denom_h + pid_i * str_denom_n_ 
    offs_tri_j = pid_i * BLOCK_C

    prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for pid_j in range(0, _n):
        offs_tri = offs_tri_j - pid_j * BLOCK_R
        offs_block = pid_j * BLOCK_R + offs_j

        affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

        # k_.matmul(_kt)
        kt_ptr = kt_j + pid_j * str_kt__n

        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + offs_k
            kc_mat = tl.load(
                kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                other=0.0
            )
            kc_mat = tl.cast(kc_mat, tl.float32)

            kt_mat = tl.load(
                kt_ptr + offs_d[:, None] * str_kt_d + offs_j[None, :] * str_kt__c, 
                mask=(offs_d[:, None] < d) & (offs_j[None, :] < BLOCK_R), 
                other=0.0
            )
            kt_mat = tl.cast(kt_mat, tl.float32)

            # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
            affinity += tl.dot(kc_mat, kt_mat, input_precision="ieee")

        # .relu()
        affinity = tl.maximum(affinity, 0.0)

        # .pow(2/3)
        affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

        # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
        dest_ptr = dest_j + pid_j * str_dest__n
        dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
        dest_mat = tl.cast(dest_mat, tl.float32)
        dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

        affinity = affinity * src_mat[:, None] * dest_mat[None, :]

        # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
        affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
        affinity = tl.log(1.0 - affinity) 

        # .triu(i*c_-j*_c+1)
        affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

        # .cumsum(3)
        curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
        affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
        prev_sum += curr_sum

        # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
        affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
        affinity = tl.cast(affinity, tl.float32)

        # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
        xq_ptr = xq_j + pid_j * str_xq__n
        denom_ptr = denom_j + offs_block * str_denom_l
        out_ptr = out_j + offs_block * str_out_l

        # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
        # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
        for rep in range(0, r):
            xq_rep_ptr = xq_ptr + rep * str_xq_r
            denom_rep_ptr = denom_ptr + rep * str_denom_r
            out_rep_ptr = out_ptr + rep * str_out_r

            kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                kc_mat = tl.load(
                    kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                kc_mat = tl.cast(kc_mat, tl.float32)

                xq_mat = tl.load(
                    xq_rep_ptr + offs_j[:, None] * str_xq__c + offs_d[None, :] * str_xq_d, 
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                    other=0.0
                )
                xq_mat = tl.cast(xq_mat, tl.float32)

                # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
                kq += tl.dot(kc_mat,tl.trans(xq_mat), input_precision="ieee")

            score = kq + affinity

            # Stabilize logsumexp using the subtract max trick
            score_max = tl.max(score, axis=0) 
            score_shifted = score - score_max[None, :]
            score_exp = tl.exp(score_shifted)
            score_sumexp = tl.sum(score_exp, axis=0)
            score_logsumexp = score_max + tl.log(score_sumexp)

            tl.store(denom_rep_ptr, score_logsumexp, mask=offs_block < BLOCK_R * _n)

            # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
            score_softmax = tl.div_rn(tl.trans(score_exp), score_sumexp[:, None])
            score_softmax = tl.cast(score_softmax, DTYPE) 
            
            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                vc_mat = tl.load(
                    vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                # vc_mat = tl.cast(vc_mat, tl.float32)
                softmax_v = tl.dot(score_softmax, vc_mat, input_precision="ieee")

                tl.store(
                    out_rep_ptr[:, None] + offs_d[None, :] * str_out_d, 
                    softmax_v, 
                    mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                )

def _universal_attention_fwd(kc, vc, xq, static_src, static_dest):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c

    Outputs:
    out: b h r l d n_
    denom: b h r l n_

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)
    '''    
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_*c_
    dtype = xq.dtype
    device = xq.device
    DTYPE_FLAG = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32

    out = torch.empty(b,h,r,l,d,n_, dtype=dtype, device=device)
    denom = torch.empty(b,h,r,l,n_, dtype=dtype, device=device)

    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c

    # Due to the sum buffer for column cumulative sum, we want to process that dimension sequencially
    grid = (b,h,n_)

    # print("Entering kernel")
    _universal_attention_fwd_kernel[grid](
        kc, vc, xq, kt, static_src, static_dest, out, denom,                                                  
        b, h, r, n_, _n, d, 
        kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3), kc.stride(4),   
        vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), vc.stride(4),   
        xq.stride(0), xq.stride(1), xq.stride(2), xq.stride(3), xq.stride(4), xq.stride(5),  
        kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3), kt.stride(4),   
        static_src.stride(0), static_src.stride(1), static_src.stride(2), static_src.stride(3), 
        static_dest.stride(0), static_dest.stride(1), static_dest.stride(2), static_dest.stride(3), 
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4), out.stride(5), 
        denom.stride(0), denom.stride(1), denom.stride(2), denom.stride(3), denom.stride(4), 
        BLOCK_R=_c, BLOCK_C=c_, BLOCK_D=64, DTYPE=DTYPE_FLAG, 
    )
    # print("Exited kernel")

    return out, denom


'''
#######################################
#     Backward Kernel & Interface     #
#######################################
'''
# @triton.autotune(
#     configs=configs,
#     key=['r', 'n_', '_n', 'd'],
# )
@triton.jit
def _universal_attention_bwd_kernel(
    # Pointers to matrices
    kc, vc, xq, kt, src, dest, dout, ddenom,    
    dkc, dvc, dxq, dsrc, ddest,
    # Matrix dimensions
    b, h, r, n_, _n, d, 
    # Strides
    str_kc_b, str_kc_h, str_kc_n_, str_kc_c_, str_kc_d,                     # b h n_ c_ d
    str_vc_b, str_vc_h, str_vc_n_, str_vc_c_, str_vc_d,                     # b h n_ c_ d
    str_xq_b, str_xq_h, str_xq_r, str_xq__n, str_xq__c, str_xq_d,           # b h r _n _c d
    str_kt_b, str_kt_h, str_kt_d, str_kt__n, str_kt__c,                     # b h d _n _c
    str_src_b, str_src_h, str_src_n_, str_src_c_,                           # b h n_ c_
    str_dest_b, str_dest_h, str_dest__n, str_dest__c,                       # b h _n _c
    str_dout_b, str_dout_h, str_dout_r, str_dout_l, str_dout_d, str_dout_n_,# b h r l d n_
    str_ddenom_b, str_ddenom_h, str_ddenom_r, str_ddenom_l, str_ddenom_n_,  # b h r l n_
    str_dkc_b, str_dkc_h, str_dkc_l, str_dkc_d,                             # b h l d
    str_dvc_b, str_dvc_h, str_dvc_n_, str_dvc_c_, str_dvc_d,                # b h n_ c_ d
    str_dxq_b, str_dxq_h, str_dxq_r, str_dxq__n, str_dxq__c, str_dxq_d,     # b h r _n _c d
    str_dsrc_b, str_dsrc_h, str_dsrc_n_, str_dsrc_c_,                       # b h n_ c_
    str_ddest_b, str_ddest_h, str_ddest__n, str_ddest__c,                   # b h _n _c
    # Meta-parameters
    BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_D: tl.constexpr,    # Block dims
    DTYPE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_i = tl.arange(0, BLOCK_C)          # c_
    offs_j = tl.arange(0, BLOCK_R)          # _c
    offs_k = tl.arange(0, BLOCK_D)          # d

    kc_i = kc + pid_b * str_kc_b + pid_h * str_kc_h
    vc_i = vc + pid_b * str_vc_b + pid_h * str_vc_h
    src_i = src + pid_b * str_src_b + pid_h * str_src_h

    # Pointers that depend on j
    xq_j = xq + pid_b * str_xq_b + pid_h * str_xq_h
    kt_j = kt + pid_b * str_kt_b + pid_h * str_kt_h 
    dest_j = dest + pid_b * str_dest_b + pid_h * str_dest_h

    dxq_j = dxq + pid_b * str_dxq_b + pid_h * str_dxq_h
    dkc_i = dkc + pid_b * str_dkc_b + pid_h * str_dkc_h 
    dvc_i = dvc + pid_b * str_dvc_b + pid_h * str_dvc_h
    dout_ij = dout + pid_b * str_dout_b + pid_h * str_dout_h
    ddenom_ij = ddenom + pid_b * str_ddenom_b + pid_h * str_ddenom_h
    dsrc_i = dsrc + pid_b * str_dsrc_b + pid_h * str_dsrc_h
    ddest_j = ddest + pid_b * str_ddest_b + pid_h * str_ddest_h

    # Clear out output storage first
    for pid_i in range(0, n_):
        dvc_ptr = dvc_i + pid_i * str_dvc_n_
        dkc_ptr = dkc_i + pid_i * BLOCK_C * str_dkc_l
        dsrc_ptr = dsrc_i + pid_i * str_dsrc_n_ 
        for d_offset in range(0, d, BLOCK_D):
            offs_d = d_offset + offs_k
            tl.store(
                dvc_ptr + offs_i[:, None] * str_dvc_c_ + offs_d[None, :] * str_dvc_d, 
                tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32), 
                mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
            )
            tl.store(
                dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32), 
                mask=(offs_i[:, None] < BLOCK_C * n_) & (offs_d[None, :] < d), 
            ) 
        tl.store(dsrc_ptr + offs_i * str_dsrc_c_, tl.zeros((BLOCK_C,), dtype=tl.float32), mask=offs_i < BLOCK_C)

    for pid_j in range(0, _n):
        dxq_ptr = dxq_j + pid_j * str_dxq__n
        ddest_ptr = ddest_j + pid_j * str_ddest__n
        for rep in range(0, r):
            dxq_rep_ptr = dxq_ptr + rep * str_dxq_r
            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                tl.store(
                    dxq_rep_ptr + offs_j[:, None] * str_dxq__c + offs_d[None, :] * str_dxq_d, 
                    tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32),
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                )
        tl.store(ddest_ptr + offs_j * str_ddest__c, tl.zeros((BLOCK_R,), dtype=tl.float32), mask=offs_j < BLOCK_R)

    for pid_i in range(0, n_):
        kc_ptr = kc_i + pid_i * str_kc_n_ 
        vc_ptr = vc_i + pid_i * str_vc_n_ 
        src_ptr = src_i + pid_i * str_src_n_ 
        src_mat = tl.load(src_ptr + offs_i * str_src_c_, mask=offs_i < BLOCK_C, other=0.0)
        src_mat = tl.cast(src_mat, tl.float32)
        src_mat = tl.exp2(tl.log2(src_mat) / 3.0)
        offs_tri_j = pid_i * BLOCK_C

        dkc_ptr = dkc_i + pid_i * BLOCK_C * str_dkc_l 
        dvc_ptr = dvc_i + pid_i * str_dvc_n_
        dout_j = dout_ij + pid_i * str_dout_n_
        ddenom_j = ddenom_ij + pid_i * str_ddenom_n_ 
        dsrc_ptr = dsrc_i + pid_i * str_dsrc_n_ 

        # Clear out sum buffers
        prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        daff_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)

        # First forward pass
        for pid_j in range(0, _n):
            offs_tri = offs_tri_j - pid_j * BLOCK_R
            offs_block = pid_j * BLOCK_R + offs_j

            affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

            # k_.matmul(_kt)
            kt_ptr = kt_j + pid_j * str_kt__n

            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                kc_mat = tl.load(
                    kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                kc_mat = tl.cast(kc_mat, tl.float32)

                kt_mat = tl.load(
                    kt_ptr + offs_d[:, None] * str_kt_d + offs_j[None, :] * str_kt__c, 
                    mask=(offs_d[:, None] < d) & (offs_j[None, :] < BLOCK_R), 
                    other=0.0
                )
                kt_mat = tl.cast(kt_mat, tl.float32)

                # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
                affinity += tl.dot(kc_mat, kt_mat, input_precision="ieee")

            # .relu()
            affinity = tl.maximum(affinity, 0.0)

            # .pow(2/3)
            affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

            # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
            dest_ptr = dest_j + pid_j * str_dest__n
            dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
            dest_mat = tl.cast(dest_mat, tl.float32)
            dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

            affinity = affinity * src_mat[:, None] * dest_mat[None, :]

            # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
            affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
            affinity = tl.log(1.0 - affinity) 

            # .triu(i*c_-j*_c+1)
            affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

            # .cumsum(3)
            curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
            affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
            prev_sum += curr_sum

            # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
            affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
            affinity = tl.cast(affinity, tl.float32)

            # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
            xq_ptr = xq_j + pid_j * str_xq__n
            ddenom_ptr = ddenom_j + offs_block * str_ddenom_l
            dout_ptr = dout_j + offs_block * str_dout_l
                        
            dxq_ptr = dxq_j + pid_j * str_dxq__n

            # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
            # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
            for rep in range(0, r):
                xq_rep_ptr = xq_ptr + rep * str_xq_r
                ddenom_rep_ptr = ddenom_ptr + rep * str_ddenom_r 
                dout_rep_ptr = dout_ptr + rep * str_dout_r

                dxq_rep_ptr = dxq_ptr + rep * str_dxq_r

                kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k
                    kc_mat = tl.load(
                        kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    kc_mat = tl.cast(kc_mat, tl.float32)

                    xq_mat = tl.load(
                        xq_rep_ptr + offs_j[:, None] * str_xq__c + offs_d[None, :] * str_xq_d, 
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    xq_mat = tl.cast(xq_mat, tl.float32)

                    # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
                    kq += tl.dot(kc_mat,tl.trans(xq_mat), input_precision="ieee")

                score = kq + affinity

                # Stabilize logsumexp using the subtract max trick
                score_max = tl.max(score, axis=0) 
                score_shifted = score - score_max[None, :]
                score_exp = tl.exp(score_shifted)
                score_sumexp = tl.sum(score_exp, axis=0)
                score_logsumexp = score_max + tl.log(score_sumexp)

                # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
                score_softmax = tl.div_rn(score_exp, score_sumexp[None, :])
                score_softmax = tl.cast(score_softmax, DTYPE) 

                # _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
                dscore_acc = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k
                    vc_mat = tl.load(
                        vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    vc_mat = tl.cast(vc_mat, tl.float32)
                    dout_mat = tl.load(
                        dout_rep_ptr[:, None] + offs_d[None, :] * str_dout_d, 
                        mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dout_mat = tl.cast(dout_mat, tl.float32)
                    dscore_acc += tl.dot(vc_mat, tl.trans(dout_mat), input_precision="ieee")
                
                # _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
                _dscore = (dscore_acc - tl.sum(dscore_acc * score_softmax, axis=0, keep_dims=True)) * score_softmax
                # _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)
                ddenom_mat = tl.load(ddenom_rep_ptr, mask=offs_block < BLOCK_R * _n, other=0.0)
                _dscore += score_softmax * ddenom_mat[None, :]
                
                # Compute the sum for the backward cumsum
                # _daff = _dscore.sum(2)  # b h c_ _c
                # daff_sum += _daff.sum(3)  # b h c_ 
                daff_sum += tl.sum(_dscore, axis=1)

                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k

                    # dvc[:,:,i] += _sscore.to(dtype=dvc.dtype).matmul(_dout_).sum(2)  # b h r c_ _c, b h r _c d -> b h c_ d
                    # sum(2) is handled via accumulating over r
                    dvc_mat = tl.load(
                        dvc_ptr + offs_i[:, None] * str_dvc_c_ + offs_d[None, :] * str_dvc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dvc_mat = tl.cast(dvc_mat, tl.float32)

                    dout_mat = tl.load(
                        dout_rep_ptr[:, None] + offs_d[None, :] * str_dout_d, 
                        mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dout_mat = tl.cast(dout_mat, tl.float32)

                    tl.store(
                        dvc_ptr + offs_i[:, None] * str_dvc_c_ + offs_d[None, :] * str_dvc_d, 
                        dvc_mat + tl.dot(score_softmax, dout_mat, input_precision="ieee"), 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    )

                    # dxq[:,:,:,j] += _dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # b h r c_ _c, b h c_ d -> b h r _c d
                    kc_mat = tl.load(
                        kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    kc_mat = tl.cast(kc_mat, tl.float32)

                    dxq_mat = tl.load(
                        dxq_rep_ptr + offs_j[:, None] * str_dxq__c + offs_d[None, :] * str_dxq_d, 
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                        other=0.0
                    )

                    tl.store(
                        dxq_rep_ptr + offs_j[:, None] * str_dxq__c + offs_d[None, :] * str_dxq_d, 
                        dxq_mat + tl.dot(tl.trans(_dscore), kc_mat, input_precision="ieee"),
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                    )
                    # @Haochen: I'm here!
                    # dkc[:,:,i] += _dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(_q.flatten(2,3))  # b h r c_ _c, b h r _c d -> b h c_ d
                    dkc_mat = tl.load(
                        dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dkc_mat = tl.cast(dkc_mat, tl.float32)

                    xq_mat = tl.load(
                        xq_rep_ptr + offs_j[:, None] * str_xq__c + offs_d[None, :] * str_xq_d, 
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    xq_mat = tl.cast(xq_mat, tl.float32)

                    tl.store(
                        dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                        dkc_mat + tl.dot(_dscore, xq_mat, input_precision="ieee"),
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    )
        # First forward pass done

        # Second forward pass with gradient computes
        prev_sum = tl.zeros((BLOCK_C,), dtype=tl.float32)
        prev_dsum = tl.zeros((BLOCK_C,), dtype=tl.float32) # sum buffer for backward 
        for pid_j in range(0, _n):
            offs_tri = offs_tri_j - pid_j * BLOCK_R
            offs_block = pid_j * BLOCK_R + offs_j
            ddest_ptr = ddest_j + pid_j * str_ddest__n

            affinity = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

            # k_.matmul(_kt)
            kt_ptr = kt_j + pid_j * str_kt__n

            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                kc_mat = tl.load(
                    kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                kc_mat = tl.cast(kc_mat, tl.float32)

                kt_mat = tl.load(
                    kt_ptr + offs_d[:, None] * str_kt_d + offs_j[None, :] * str_kt__c, 
                    mask=(offs_d[:, None] < d) & (offs_j[None, :] < BLOCK_R), 
                    other=0.0
                )
                kt_mat = tl.cast(kt_mat, tl.float32)

                # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
                affinity += tl.dot(kc_mat, kt_mat, input_precision="ieee")

            _aff1 = affinity

            # .relu()
            affinity = tl.maximum(affinity, 0.0)

            # .pow(2/3)
            affinity = tl.exp2(tl.log2(affinity) * 2.0 / 3.0)

            # * static_src_.pow(1/3).unsqueeze(-1) * static_dest.pow(1/3).unsqueeze(-2)
            dest_ptr = dest_j + pid_j * str_dest__n
            dest_mat = tl.load(dest_ptr + offs_j * str_dest__c, mask=offs_j < BLOCK_R, other=0.0)
            dest_mat = tl.cast(dest_mat, tl.float32)
            dest_mat = tl.exp2(tl.log2(dest_mat) / 3.0)

            affinity = affinity * src_mat[:, None] * dest_mat[None, :]

            _aff2 = affinity

            # torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())
            affinity = tl.clamp(affinity, 0.0, 1.0 - 1e-6)
            affinity = tl.log(1.0 - affinity) 

            # .triu(i*c_-j*_c+1)
            affinity = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), affinity, 0.0)

            # .cumsum(3)
            curr_sum = tl.sum(affinity, axis=1, keep_dims=False) 
            affinity = tl.cumsum(affinity, axis=1) + prev_sum[:, None]  
            prev_sum += curr_sum

            # .masked_fill(mask.tril(i*c_-j*_c-1), -1e12)
            affinity = tl.where((offs_j[None, :] < (offs_i[:, None] + offs_tri)), -1e12, affinity)
            affinity = tl.cast(affinity, tl.float32)

            # k_.unsqueeze(2).matmul(_q.transpose(-1,-2)).add(affinity.unsqueeze(2))
            xq_ptr = xq_j + pid_j * str_xq__n
            ddenom_ptr = ddenom_j + offs_block * str_ddenom_l
            dout_ptr = dout_j + offs_block * str_dout_l
                        
            dxq_ptr = dxq_j + pid_j * str_dxq__n

            _daff = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)

            # @Haochen: storing a 3D tensor after applying .dot() to 3D tensors would fail LLVM compilation
            # The problem is fixed in triton 3.2.0, and the alternative code is listed in matmul.py
            for rep in range(0, r):
                xq_rep_ptr = xq_ptr + rep * str_xq_r
                ddenom_rep_ptr = ddenom_ptr + rep * str_ddenom_r 
                dout_rep_ptr = dout_ptr + rep * str_dout_r

                dxq_rep_ptr = dxq_ptr + rep * str_dxq_r

                kq = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k
                    kc_mat = tl.load(
                        kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    kc_mat = tl.cast(kc_mat, tl.float32)

                    xq_mat = tl.load(
                        xq_rep_ptr + offs_j[:, None] * str_xq__c + offs_d[None, :] * str_xq_d, 
                        mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    xq_mat = tl.cast(xq_mat, tl.float32)

                    # Use ieee to use fp32, otherwise the default would be tf32 even after tl.cast
                    kq += tl.dot(kc_mat,tl.trans(xq_mat), input_precision="ieee")

                score = kq + affinity

                # Stabilize logsumexp using the subtract max trick
                score_max = tl.max(score, axis=0) 
                score_shifted = score - score_max[None, :]
                score_exp = tl.exp(score_shifted)
                score_sumexp = tl.sum(score_exp, axis=0)
                score_logsumexp = score_max + tl.log(score_sumexp)

                # score.transpose(-1,-2).softmax(dim=-1).to(dtype=_q.dtype).matmul(v_.unsqueeze(2))
                score_softmax = tl.div_rn(score_exp, score_sumexp[None, :])
                score_softmax = tl.cast(score_softmax, DTYPE) 

                # _dscore = v_.unsqueeze(2).matmul(_dout_.transpose(-1,-2))  # b h c_ d, b h r _c d -> b h r c_ _c  (from out)
                dscore_acc = tl.zeros((BLOCK_C, BLOCK_R), dtype=tl.float32)
                for d_offset in range(0, d, BLOCK_D):
                    offs_d = d_offset + offs_k
                    vc_mat = tl.load(
                        vc_ptr + offs_i[:, None] * str_vc_c_ + offs_d[None, :] * str_vc_d, 
                        mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    vc_mat = tl.cast(vc_mat, tl.float32)
                    dout_mat = tl.load(
                        dout_rep_ptr[:, None] + offs_d[None, :] * str_dout_d, 
                        mask=(offs_block[:, None] < BLOCK_R * _n) & (offs_d[None, :] < d), 
                        other=0.0
                    )
                    dout_mat = tl.cast(dout_mat, tl.float32)
                    dscore_acc += tl.dot(vc_mat, tl.trans(dout_mat), input_precision="ieee")
                
                # _dscore = _dscore.sub(_dscore.mul(_sscore).sum(-2,True)).mul(_sscore)  # (from softmax)
                _dscore = (dscore_acc - tl.sum(dscore_acc * score_softmax, axis=0, keep_dims=True)) * score_softmax
                # _dscore += _sscore * _ddenom_.unsqueeze(-2)  # (from denom)
                ddenom_mat = tl.load(ddenom_rep_ptr, mask=offs_block < BLOCK_R * _n, other=0.0)
                _dscore += score_softmax * ddenom_mat[None, :]
                
                # Compute the sum for the backward cumsum
                # _daff = _dscore.sum(2)  # b h c_ _c
                _daff += _dscore
            
            # _daff_cs = _daff.cumsum(3)  # (from cumsum)
            # _daff_cs += dsum_buffer.unsqueeze(-1)   # Accumulate across row chunks
            # dsum_buffer = _daff_cs[:,:,:,-1].clone()  # Cloning prevents subsequent _aff2 calcs from changing buffer values                
            curr_dsum = tl.sum(_daff, axis=1, keep_dims=False) 
            _daff_cs = tl.cumsum(_daff, axis=1) + prev_dsum[:, None]  
            prev_dsum += curr_dsum

            # _daff += daff_sum.unsqueeze(-1) - _daff_cs
            _daff = daff_sum[:, None] - _daff_cs + _daff

            # _daff = _daff.triu(i*c_-j*_c+1)  # Prevent accumulation into masked regions
            _daff = tl.where((offs_j[None, :] > (offs_i[:, None] + offs_tri)), _daff, 0.0)

            # _daff /= _aff2.clamp(min=1e-6, max=1-1e-6) - 1  # ( from ln(1-x) )
            _daff = tl.div_rn(_daff, (tl.clamp(_aff2, 1e-6, 1.0 - 1e-6) - 1.0))

            # _daff *= _aff2.le(1-1e-6)
            _daff = tl.where((_aff2 <= (1.0 - 1e-6)), _daff, 0.0)

            # _dstat = _daff.mul(_aff1.relu().pow(2/3)).to(dtype=static_src.dtype)  # b h c_ _c
            _dstat = _daff * tl.exp2(tl.log2(tl.maximum(_aff1, 0.0)) * 2.0 / 3.0)
            _dstat = tl.cast(_dstat, tl.float32)

            # Backprop into stat_src and stat_dest            
            # dstat_src[:,:,i] += _dstat.mul(_static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # b h c_ _c, b h _c -> b h c_
            dsrc_mat = tl.load(dsrc_ptr + offs_i * str_dsrc_c_, mask=offs_i < BLOCK_C, other=0.0)
            dsrc_mat += tl.div_rn(tl.sum(_dstat * dest_mat[None, :], axis=1), tl.exp2(tl.log2(src_mat) * 2.0) * 3.0)
            tl.store(dsrc_ptr + offs_i * str_dsrc_c_, dsrc_mat, mask=offs_i < BLOCK_C)

            # dstat_dest[:,:,j] += _dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(_static_dest.pow(2).mul(3))  # b h c_ _c, b h c_ -> b h _c
            ddest_mat = tl.load(ddest_ptr + offs_j * str_ddest__c, mask=offs_j < BLOCK_R, other=0.0)
            ddest_mat += tl.div_rn(tl.sum(_dstat * src_mat[:, None], axis=0), tl.exp2(tl.log2(dest_mat) * 2.0) * 3.0)
            tl.store(ddest_ptr + offs_j * str_ddest__c, ddest_mat, mask=offs_j < BLOCK_R)

            # # Backprop into k/k matmul
            # _daff *= static_src_.unsqueeze(-1) * _static_dest.unsqueeze(-2)  # (from prod with statics)
            _daff = _daff * src_mat[:, None] * dest_mat[None, :]
            
            # _daff = _daff.to(dtype=_q.dtype) * _aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(_aff1.gt(0))  # (from relu and pow)
            _daff = tl.cast(_daff, tl.float32) * tl.exp2( - tl.log2((tl.abs(_aff1) + 1e-9)) / 3.0) * (2.0 / 3.0)
            _daff = tl.where((_aff1 > 0.0), _daff, 0.0)

            for d_offset in range(0, d, BLOCK_D):
                offs_d = d_offset + offs_k
                dkc_j_ptr = dkc_i + pid_j * BLOCK_R * str_dkc_l 

                # dkc[:,:,j*_c:(j+1)*_c] += _daff.transpose(-1,-2).matmul(k_)  # b h c_ _c, b h c_ d -> b h _c d
                dkc_j_mat = tl.load(
                    dkc_j_ptr + offs_j[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                    other=0.0
                )
                dkc_j_mat = tl.cast(dkc_j_mat, tl.float32)
                
                kc_mat = tl.load(
                    kc_ptr + offs_i[:, None] * str_kc_c_ + offs_d[None, :] * str_kc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                kc_mat = tl.cast(kc_mat, tl.float32)

                tl.store(
                    dkc_j_ptr + offs_j[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    dkc_j_mat + tl.dot(tl.trans(_daff), kc_mat, input_precision="ieee"),
                    mask=(offs_j[:, None] < BLOCK_R) & (offs_d[None, :] < d), 
                )
                
                # dkc[:,:,i*c_:(i+1)*c_] += _daff.matmul(_kt.transpose(-1,-2))  # b h c_ _c, b h d _c -> b h c_ d
                dkc_mat = tl.load(
                    dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                    other=0.0
                )
                dkc_mat = tl.cast(dkc_mat, tl.float32)

                kt_mat = tl.load(
                    kt_ptr + offs_d[:, None] * str_kt_d + offs_j[None, :] * str_kt__c, 
                    mask=(offs_d[:, None] < d) & (offs_j[None, :] < BLOCK_R), 
                    other=0.0
                )
                kt_mat = tl.cast(kt_mat, tl.float32)

                tl.store(
                    dkc_ptr + offs_i[:, None] * str_dkc_l + offs_d[None, :] * str_dkc_d, 
                    dkc_mat + tl.dot(_daff, tl.trans(kt_mat), input_precision="ieee"),
                    mask=(offs_i[:, None] < BLOCK_C) & (offs_d[None, :] < d), 
                )



def _universal_attention_bwd(kc, vc, xq, static_src, static_dest, dout, ddenom):
    '''
    Inputs:
    kc: b h n_ c_ d
    vc: b h n_ c_ d
    xq: b h r _n _c d
    static_src: b h n_ c_
    static_dest: b h _n _c
    dout: b h r l d n_
    ddenom: b h r l n_

    Outputs:
    dkc: b h n_ c_ d
    dvc: b h n_ c_ d
    dxq: b h r _n _c d
    dstatic_src: b h n_ c_
    dstatic_dest: b h _n _c

    Intermediate: 
    variables_ are split over cols
    _variables are split over rows
    (i.e. n_ is the number of col chunks, _n is the number of row chunks)

    Note: when using mixed precision, dout is downcasted but ddenom is always fp32
    '''
    b,h,r,_n,_c,d = xq.shape
    _,_,n_,c_,_ = kc.shape
    l = n_*c_
    dtype = xq.dtype
    device = xq.device
    DTYPE_FLAG = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32

    dkc = torch.empty(b,h,l,d, dtype=dtype, device=device)
    dvc = torch.empty(b,h,n_,c_,d, dtype=dtype, device=device)
    dxq = torch.empty(b,h,r,_n,_c,d, dtype=dtype, device=device)
    dstatic_src = torch.empty(b,h,n_,c_, dtype=static_src.dtype, device=device)
    dstatic_dest = torch.empty(b,h,_n,_c, dtype=dtype, device=device)

    kt = kc.view(b,h,_n,_c,d).permute(0,1,4,2,3)  # b h d _n _c
    # The total size of the buffers are O(n^2), so it would probably better if we can avoid using them
    # sum_buffer = torch.empty(b,h,n_,c_, dtype=static_src.dtype, device=static_src.device)
    # aff1 = torch.empty(b,h,n_,c_,l, dtype=dtype, device=device)
    # sscore = torch.empty(b,h,r,n_,c_,l, dtype=dtype, device=device)

    # Due to the sum buffer for column cumulative sum, we want to process that dimension sequencially
    # And since the backward pass needs accumulation on the rows, we want to process that dimension sequencially as well
    grid = (b,h)

    _universal_attention_bwd_kernel[grid](
        kc, vc, xq, kt, static_src, static_dest, dout, ddenom,
        dkc, dvc, dxq, dstatic_src, dstatic_dest,
        b, h, r, n_, _n, d, 
        kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3), kc.stride(4),  
        vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3), vc.stride(4),   
        xq.stride(0), xq.stride(1), xq.stride(2), xq.stride(3), xq.stride(4), xq.stride(5),  
        kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3), kt.stride(4),   
        static_src.stride(0), static_src.stride(1), static_src.stride(2), static_src.stride(3), 
        static_dest.stride(0), static_dest.stride(1), static_dest.stride(2), static_dest.stride(3), 
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3), dout.stride(4), dout.stride(5), 
        ddenom.stride(0), ddenom.stride(1), ddenom.stride(2), ddenom.stride(3), ddenom.stride(4), 
        dkc.stride(0), dkc.stride(1), dkc.stride(2), dkc.stride(3),
        dvc.stride(0), dvc.stride(1), dvc.stride(2), dvc.stride(3), dvc.stride(4), 
        dxq.stride(0), dxq.stride(1), dxq.stride(2), dxq.stride(3), dxq.stride(4), dxq.stride(5), 
        dstatic_src.stride(0), dstatic_src.stride(1), dstatic_src.stride(2), dstatic_src.stride(3), 
        dstatic_dest.stride(0), dstatic_dest.stride(1), dstatic_dest.stride(2), dstatic_dest.stride(3), 
        BLOCK_R=_c, BLOCK_C=c_, BLOCK_D=64, DTYPE=DTYPE_FLAG, 
    )
    dkc = dkc.view(b,h,n_,c_,d)

    return dkc, dvc, dxq, dstatic_src, dstatic_dest












class UniversalAttention(Function):
    @staticmethod
    def forward(kc, vc, xq, static_src, static_dest):
        out, denom = _universal_attention_fwd(kc, vc, xq, static_src, static_dest)
        # b,h,r,l,d = xq.shape
        # _,_,n,c,_ = kc.shape
        # mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
        # out = torch.empty(b,h,r,l,d,n, dtype=xq.dtype, device=xq.device)
        # denom = torch.empty(b,h,r,l,n, dtype=xq.dtype, device=xq.device)
        # affs = torch.empty(b,h,l, dtype=torch.float, device=xq.device)
        # static_src = static_src.pow(1/3)
        # static_dest = static_dest.pow(1/3)
        # kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l
        # for i in range(n):
        #     k_ = kc[:,:,i]  # b h c d
        #     v_ = vc[:,:,i]  # b h c d
        #     static_src_ = static_src[:,:,i]  # b h c

        #     # Calculate decay matrix
        #     affinity = k_.matmul(kt).relu().pow(2/3).float()  # deltanet style decay
        #     affinity = affinity * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)  # incorporate mamba-style and per-token decay
        #     affinity = torch.log1p(affinity.clamp(min=0, max=1-1e-6).neg())  # b h c l
        #     affinity = affinity.triu(i*c+1).cumsum(3)  # Accumulate decay with causal masking
        #     affinity = affinity.masked_fill(mask.tril(i*c-1), -1e12)  # Re-mask, with 1s on diagonal

        #     # Perform actual attention operation
        #     score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(affinity.unsqueeze(2))  # b h r c l
        #     denom_ = score.logsumexp(dim=-2)  # b h r l
        #     out_ = score.transpose(-1,-2).softmax(dim=-1).to(dtype=xq.dtype).matmul(v_.unsqueeze(2))  # b h r l d

        #     out[...,i] = out_
        #     denom[...,i] = denom_
        #     affs[...,i*c:(i+1)*c] = affinity[...,-1]
        return out, denom, None

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        kc,vc,xq,ss,sd = inputs
        # out,denom = outputs
        ctx.save_for_backward(kc,vc,xq,ss,sd)

    @staticmethod
    def backward(ctx, g_out, g_denom, g_affs):
        kc,vc,xq,static_src,static_dest = ctx.saved_tensors
        dkc,dvc,dxq,dstat_src,dstat_dest = _universal_attention_bwd(kc, vc, xq, static_src, static_dest, g_out, g_denom)
        
        # # Note: when using mixed precision, g_out is downcast but g_denom is always fp32
        # kc,vc,xq,static_src,static_dest = ctx.saved_tensors
        # dkc,dvc,dxq,dstat_src,dstat_dest = [torch.zeros_like(x) for x in [kc,vc,xq,static_src,static_dest]]
        # b,h,r,l,d = xq.shape
        # _,_,n,c,_ = kc.shape
        # mask = torch.ones(c,l, dtype=torch.bool, device=xq.device)
        # static_src = static_src.pow(1/3)
        # static_dest = static_dest.pow(1/3)
        # kt = kc.view(b,h,l,d).transpose(-2,-1)  # b h d l

        # for i in range(n):
        #     k_ = kc[:,:,i]  # b h c d
        #     v_ = vc[:,:,i]  # b h c d
        #     static_src_ = static_src[:,:,i]  # b h c
        #     dout_ = g_out[...,i]
        #     ddenom_ = g_denom[...,i]

        #     # Rerun forward pass
        #     aff1 = k_.matmul(kt)
        #     aff2 = aff1.relu().pow(2/3).float()
        #     aff3 = aff2 * static_src_.unsqueeze(-1) * static_dest.unsqueeze(-2)
        #     score = torch.log1p(aff3.clamp(min=0,max=1-1e-6).neg()).triu(i*c+1).cumsum(3).masked_fill(mask.tril(i*c-1), -1e12)
        #     score = k_.unsqueeze(2).matmul(xq.transpose(-1,-2)).add(score.unsqueeze(2))  # b h r c l
        #     sscore = score.softmax(dim=-2)

        #     # Backward pass
        #     dvc[:,:,i] += sscore.to(dtype=dvc.dtype).matmul(dout_).sum(2)  # bhrcl,bhrld -> bhcd
            
        #     dscore = v_.unsqueeze(2).matmul(dout_.transpose(-1,-2))  # bhcd,bhrld -> bhrcl   <-- from out
        #     dscore = dscore.sub(dscore.mul(sscore).sum(-2,True)).mul(sscore)  # <-- from softmax
        #     dscore += sscore * ddenom_.unsqueeze(-2)  # b h r c l   <-- from denom

        #     dxq += dscore.to(dtype=dxq.dtype).transpose(-1,-2).matmul(k_.unsqueeze(2))  # bhrcl, bhcd -> bhrld
        #     dkc[:,:,i] += dscore.to(dtype=dkc.dtype).transpose(2,3).flatten(3,4).matmul(xq.flatten(2,3))  # bhrcl, bhrld -> bhcd

        #     daff = dscore.sum(2)  # b h c l
        #     daff = daff.flip([3]).cumsum(3).flip([3]).triu(i*c+1)  # <-- from cumsum
        #     daff /= aff3.clamp(min=1e-6, max=1-1e-6)-1  # <-- from ln(1-x)
        #     daff *= aff3.ge(0)
        #     daff *= aff3.le(1-1e-6)
        #     dstat = daff.mul(aff2).to(dtype=static_src.dtype)  # b h c l

        #     dstat_src[:,:,i] += dstat.mul(static_dest.unsqueeze(-2)).sum(-1).div(static_src_.pow(2).mul(3))  # bhcl, bhl -> bhc
        #     dstat_dest += dstat.mul(static_src_.unsqueeze(-1)).sum(-2).div(static_dest.pow(2).mul(3))  # bhcl, bhc -> bhl

        #     daff = daff.mul(static_src_.unsqueeze(-1)*static_dest.unsqueeze(-2))  # <-- from prod with statics
        #     daff = daff.to(dtype=xq.dtype) * aff1.abs().add(1e-9).pow(-1/3).mul(2/3).mul(aff1.gt(0))  # <-- from relu + pow

        #     dkc += daff.transpose(-1,-2).matmul(k_).view(b,h,n,c,d)  # bhcl, bhcd -> bhld, <-- grad via kt
        #     dkc[:,:,i] += daff.matmul(kt.transpose(-1,-2))  # bhcl, bhdl -> bhcd, <-- grad via k_

        return dkc,dvc,dxq,dstat_src,dstat_dest


class SMVecMatMul(Function):
    @staticmethod
    def forward(mat, vec):
        # mat: ... d n
        # vec: ... n
        return mat.mul(vec.softmax(dim=-1).unsqueeze(-2)).sum(-1)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        mat, vec = inputs
        ctx.save_for_backward(mat, vec)

    @ staticmethod
    def backward(ctx, g):
        mat, vec = ctx.saved_tensors
        vec = vec.softmax(dim=-1)
        d_mat = g.unsqueeze(-1).mul(vec.unsqueeze(-2))  # ... d n
        d_vec = g.unsqueeze(-1).mul(mat).sum(-2)  # ... n
        d_vec = d_vec.sub(d_vec.mul(vec).sum(-1,True)).mul(vec)
        return d_mat, d_vec


class AttentionKwargs(TypedDict, total=False):
    """
    The attention kwargs to be passed to fms model forward.

    attn_name: str
        this is the name corresponding to the attention op registered in register_attention_op
    """

    attn_name: str


class SDPAAttentionKwargs(AttentionKwargs):
    mask: NotRequired[torch.Tensor]
    attn_algorithm: NotRequired[str]
    is_causal_mask: bool


def _sdpa_update_attn_kwargs(**attn_kwargs):
    # this is updating the mask for decoding
    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.zeros(mask.size(0), 1, 1, device=mask.device),
            ),
            dim=2,
        )
        if torch._dynamo.config.dynamic_shapes:
            torch._dynamo.mark_dynamic(mask, 2)

        attn_kwargs["mask"] = mask
    return attn_kwargs


class QKV(nn.Module, metaclass=abc.ABCMeta):
    """Simple module for applying qkv in attention"""

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.emb_dim = emb_dim
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_kq_per_head = emb_kq_per_head
        self.emb_v_per_head = emb_v_per_head
        self.use_bias = use_bias
        self.linear_config = linear_config

    @abc.abstractmethod
    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """applies query/key/value transformations on q, k, v inputs respectively and returns the resulting values

        Args:
            q: torch.Tensor
                the query tensor
            k: Optional[torch.Tensor]
                the optional key tensor
            v: Optional[torch.Tensor]
                the optional value tensor

        Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            the query, key, and value computed
        """
        pass

    @abc.abstractmethod
    def reset_parameters(self):
        """resets the query, key, and value weights for training

        Args:
            gain: int
                gain for std in norm (default is 1)
        """
        pass


class UnfusedQKV(QKV):
    """
    Unfused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )

        self.query = get_linear(
            self.emb_dim,
            self.nheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.key = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_kq_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )
        self.value = get_linear(
            self.emb_dim,
            self.kvheads * self.emb_v_per_head,
            bias=use_bias,
            linear_config=linear_config,
        )

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k is None and v is None:
            k = q
            v = q
        elif k is None or v is None:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        # b x h x qlen x ds
        queries = self.query(q)
        keys = self.key(k)
        values = self.value(v)
        return queries, keys, values


class FusedQKV(QKV):
    """
    Fused Weights implementation of QKV
    """

    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        kvheads: int,
        emb_kq_per_head: int,
        emb_v_per_head: int,
        use_bias: bool,
        linear_config: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            emb_dim,
            nheads,
            kvheads,
            emb_kq_per_head,
            emb_v_per_head,
            use_bias,
            linear_config,
            *args,
            **kwargs,
        )
        self.splits = [
            self.nheads * self.emb_kq_per_head,
            self.kvheads * self.emb_kq_per_head,
            self.kvheads * self.emb_v_per_head,
        ]

        self.qkv_fused = get_linear(
            self.emb_dim,
            sum(self.splits),
            bias=self.use_bias,
            linear_config=linear_config,
        )

    def unfuse_weights(self):
        with torch.device("meta"):
            result = UnfusedQKV(
                self.emb_dim,
                self.nheads,
                self.kvheads,
                self.emb_kq_per_head,
                self.emb_v_per_head,
                self.use_bias,
            )
        query, key, value = torch.split(self.qkv_fused.weight, self.splits, dim=0)
        result.query.weight = torch.nn.Parameter(query)
        result.key.weight = torch.nn.Parameter(key)
        result.value.weight = torch.nn.Parameter(value)
        if self.use_bias:
            query_bias, key_bias, value_bias = torch.split(
                self.qkv_fused.bias, self.splits, dim=0
            )
            result.query.bias = torch.nn.Parameter(query_bias)
            result.key.bias = torch.nn.Parameter(key_bias)
            result.value.bias = torch.nn.Parameter(value_bias)
        return result

    def reset_parameters(self):
        nn.init.trunc_normal_(self.qkv_fused.weight, mean=0.0, std=0.02)
        if self.use_bias:
            self.qkv_fused.bias.data.zero_()

    def forward(
        self, q: torch.Tensor, k: Optional[torch.Tensor], v: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (k is None and v is None) or (k is q and v is q):
            qkv = q
        else:
            raise ValueError("q, k, and v must be the same or k and v must be None")
        return self.qkv_fused(qkv).split(self.splits, dim=-1)


class MultiHeadAttention(nn.Module):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    ...
    Args
    ----
    emb_dim : int
        Latent dimensionality of input and output tensors.
    emb_kq : int
        Latent dimensionality of each head in key and query projections (attention dimension).
    emb_v : int
        Latent dimensionality of each head in value projection (mixing dimension).
    nheads : int
        Number of attention heads.
    p_dropout : float|None
        Dropout probability. Must be in range [0,1]. If 0 or None, dropout will not be used.
    use_bias : bool
        Include bias terms in fully-connected sublayers?
    fused : bool
        If True, qkv weights will be fused, otherwise qkv weights will be unfused.
    linear_config : Mapping[str, Any] | None
        Configuration for selection of linear modules (QKV, dense).
        Pass as {"linear_type": [str | callable], <other kwargs>}.
        "linear_type" should provide the string identifier of a registered type
        (e.g., "torch_linear", "gptq", ...) or a callable for module selection depending
        on module name. Additional config options should be provided as kwargs in
        linear_config.
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):
        super(MultiHeadAttention, self).__init__()
        self.nheads = nheads
        self.kvheads = kvheads
        self.emb_dim = emb_dim
        self.emb_kq_per_head = emb_kq
        self.emb_v_per_head = emb_v
        self.p_dropout = p_dropout if p_dropout is not None else 0.0
        self.use_bias = use_bias
        self.fused = fused
        self.linear_config = linear_config
        self.scale_factor = scale_factor

        self.in_proj: QKV = (FusedQKV if self.fused else UnfusedQKV)(
            self.emb_dim,
            self.nheads,
            self.kvheads,
            self.emb_kq_per_head,
            self.emb_v_per_head,
            self.use_bias,
            linear_config=linear_config,
        )

        self.dense = nn.Linear(
            self.nheads * self.emb_v_per_head,
            self.emb_dim,
            bias=use_bias,
            # linear_config=linear_config,
        )

        if self.p_dropout:
            self.attn_dropout = nn.Dropout(self.p_dropout)
        self.position_encoder = position_encoder

        self.wstatic = nn.Linear(self.emb_dim, self.kvheads*2, bias=True)
        self.register_buffer("staticb", torch.empty(self.kvheads*2))

        self.UA = UniversalAttention.apply
        self.SMVMM = SMVecMatMul.apply

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                if self.use_bias:
                    m.bias.data.zero_()
            elif isinstance(m, QKV):
                m.reset_parameters()
        static_max = math.log(.1)
        static_min = math.log(.001)
        # nn.init.uniform_(self.wstatic.bias)
        self.wstatic.bias.data.zero_()
        self.staticb = torch.rand_like(self.staticb) * (static_max - static_min) + static_min

    # def to_tp(self, group: ProcessGroup) -> "TPMultiHeadAttention":
    #     return TPMultiHeadAttention.import_module(self, group)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        past_key_value_state: tuple
            the cache to be used in attention of the form (<self/cross>_key, <self/cross>_value)
        position_ids: Optional[torch.LongTensor]
            The position of each of the tokens encoded in q and k. Used for RoPE embeddings
        use_cache: bool
            if True, the kv states for self/cross attention will be saved, otherwise they will not be saved

        Returns
        -------
        tensor or tuple
            If use_cache=False, only the hidden state will be returned as a tensor. If use_cache=True, a tuple will be
            returned in the form (hidden_state, cache) where hidden_state is a tensor and cache is of the form specified
            in past_key_value_state
        """
        # q, k, v: batch_size x seq_len x emb_dim
        # mask: batch_size x seq_len x seq_len
        batch_size, q_len, _ = q.size()

        # if this is self attention, we always recompute
        # cross attention only gets computed when a cache does not exist
        # if we dont have the cache yet, we need to compute
        # d x (h x ds)
        # b x kvlen x d
        # b x kvlen x h x ds
        # b x h x kvlen x ds
        # todo: Cross attention (This always is true for now)
        q_out, k_out, v_out = self.in_proj(q, k, v)
        static = F.linear(q, self.wstatic.weight, self.staticb + self.wstatic.bias * math.sqrt(self.emb_dim))
        static = static.sigmoid().view(batch_size, q_len, 2, self.kvheads).permute(2,0,3,1)  # 2 b h l
        static_src = static[0]  # b h l
        static_dest = static[1]  # b h l

        # note: transposes will be moved in a later PR to fix dis-contiguous tensor issues
        queries = q_out.view(batch_size, q_len, self.nheads, self.emb_kq_per_head)
        keys = k_out.view(batch_size, q_len, self.kvheads, self.emb_kq_per_head)
        values = v_out.view(batch_size, q_len, self.kvheads, self.emb_v_per_head)

        # Normalize keys
        keys = keys / keys.pow(2).sum(-1, True).sqrt().add(1e-6)

        # You want to apply rotary embeddings pre-cache
        if self.position_encoder is not None:
            queries, keys = self.position_encoder.adjusted_qk(
                queries, keys, position_ids, past_key_value_state, use_cache
            )

        if past_key_value_state is not None:
            # Iterative universal attention
            assert q_len == 1, "UA decoding not currently supported for more than 1 token"
            # thresh = math.log(1e-4)
            (k, v, r, a) = past_key_value_state  # bhld, bhld, bhl, bhl
            k_ = keys.squeeze(1)  # b h d
            v_ = values.squeeze(1)  # b h d
            q = queries.view(batch_size, self.kvheads, -1, self.emb_kq_per_head)  # b h r d
            static_src = static_src.squeeze(2)  # b h
            static_dest = static_dest.squeeze(2)  # b h

            # Update k/v cache
            k = torch.cat((k, k_.unsqueeze(2)), dim=2)
            v = torch.cat((v, v_.unsqueeze(2)), dim=2)

            # q/k/k products
            qkkk = k.matmul(torch.cat([q,k_.unsqueeze(2)], dim=2).transpose(-1,-2))  # b h l+1 r+1
            qk = qkkk[...,:-1]  # b h l+1 r
            kk = qkkk[:,:,:-1,-1]  # b h l

            # Calculate decays
            decay = kk.relu().float().pow(2)
            decay = (decay * r * static_dest.unsqueeze(-1)).pow(1/3)
            decay = torch.log1p(decay.clamp(min=0, max=1-1e-6).neg())
            a = a + decay

            # Update r/a cache
            r = torch.cat((r, static_src.unsqueeze(2)), dim=2)
            a = torch.cat((a, torch.zeros(batch_size, self.kvheads, 1, device=a.device, dtype=a.dtype)), dim=2)

            # Perform scaled attention
            attn = qk.add(a.unsqueeze(-1)).softmax(dim=2).transpose(-1,-2).matmul(v)  # b h r d
            (keys, values, rates, affs) = k,v,r,a
            
        else:
            # Blockwise universal attention
            queries = queries.transpose(1,2).view(batch_size, self.kvheads, -1, q_len, self.emb_kq_per_head)  # b h r l d
            keys = keys.transpose(1,2)  # b h l d
            values = values.transpose(1,2)  # b h l d
            rates = static_src

            c = 128
            b = batch_size
            # Right-pad k,v,src if len not divisible by chunksize
            if q_len % c != 0:
                slack = c - q_len % c
                queries = torch.cat([queries, torch.zeros(b, self.kvheads, self.nheads//self.kvheads, slack, self.emb_kq_per_head, 
                                                          device=queries.device, dtype=queries.dtype)], dim=-2)
                keys = torch.cat([keys, torch.zeros(b, self.kvheads, slack, self.emb_kq_per_head, 
                                                          device=keys.device, dtype=keys.dtype)], dim=-2)
                values = torch.cat([values, torch.zeros(b, self.kvheads, slack, self.emb_v_per_head, 
                                                          device=values.device, dtype=values.dtype)], dim=-2)
                static_src = torch.cat([static_src, torch.zeros(b, self.kvheads, slack,
                                                               device=static_src.device, dtype=static_src.dtype)], dim=-1)
                static_dest = torch.cat([static_dest, torch.zeros(b, self.kvheads, slack,
                                                               device=static_dest.device, dtype=static_dest.dtype)], dim=-1)

            # Chunk inputs
            l = static_src.size(2)
            n = l//c
            s = [b, self.kvheads, n, c, -1]
            kc = keys.view(*s)  # b h n c d
            vc = values.view(*s)
            static_src = static_src.view(b, self.kvheads, n, c)  # b h n c
            static_dest = static_dest.view(b, self.kvheads, n, c)  # b h n c
            queries = queries.view(b, self.kvheads, self.nheads//self.kvheads, n, c, -1)
                    
            # Inputs:
            # kc: b h n_ c_ d
            # vc: b h n_ c_ d
            # xq: b h r _n _c d
            # static_src: b h n_ c_
            # static_dest: b h _n _c

            # Perform UA
            output, denom, affs = self.UA(kc, vc, queries, static_src, static_dest)

            # Weighted avg for final softmax
            output = self.SMVMM(output, denom)  # b h r l d
            attn = output.permute(0,3,1,2,4).reshape(b,l,-1)

            # Prune any right-padding
            keys = keys[:,:,:q_len]
            values = values[:,:,:q_len]
            affs = affs if affs is None else affs[:,:,:q_len]
            attn = attn[:,:q_len]

        attn = attn.view(batch_size, q_len, self.nheads * self.emb_v_per_head)
        out = self.dense(attn)

        # if use_cache=True, we return the hidden_state as well as the kv cache
        if use_cache:
            return out, (keys, values, rates, affs)
        else:
            return out


class TPMultiHeadAttention(MultiHeadAttention, TPModule):
    """
    Performs multi-headed self- or cross-attention, with optional attention masking.
    This subclass adds support for Tensor Parallel
    ...
    Args
    ----
    Check MultiHeadAttention for up-to-date docs

    world_size: int
        the number of processes running this model in TP
    rank: int
        the index of this process wrt to the rest running the model in TP
    """

    def __init__(
        self,
        emb_dim,
        emb_kq,
        emb_v,
        nheads,
        kvheads,
        p_dropout=None,
        use_bias=False,
        position_encoder: Optional[PositionEncoder] = None,
        fused: bool = True,
        group: Optional[ProcessGroup] = None,
        linear_config: Optional[Mapping[str, Any]] = None,
        scale_factor: Optional[float] = None,
    ):
        assert torch.distributed.is_initialized()

        rank, world_size = distributed.rank_and_world(group)
        assert nheads % world_size == 0, (
            "The number of heads must be divisible by world size"
        )
        assert (kvheads >= world_size and kvheads % world_size == 0) or (
            kvheads < world_size and world_size % kvheads == 0
        ), (
            "the kv heads must be divisible by the world size or the world size must be divisible by kv heads"
        )
        MultiHeadAttention.__init__(
            self,
            emb_dim,
            emb_kq,
            emb_v,
            nheads // world_size,
            (kvheads // world_size) if kvheads >= world_size else 1,
            p_dropout,
            use_bias,
            position_encoder,
            fused,
            linear_config,
            scale_factor,
        )
        self.pre_tp_nheads = nheads
        self.pre_tp_kvheads = kvheads
        self.setup_tp(rank, group)

        # linear_type must handle module_name = None to support TP of MHA
        self.linear_type = get_linear_type(self.linear_config)

    def load_weights(
        self,
        tensor_values: dict[str, torch.Tensor],
    ) -> Optional[set]:
        """Define sharding info of MHA module as:
        {'module_name': (module_obj, sharding_dim, max_partition)}
        Then, call the pre-registered sharding function associated with
        self.linear_type.

        `sharding_dim` is sharding dimension of the `weights` parameter
        of nn.Linear. It may differ for other types of linear or other
        parameters.

        The numbers in `max_partition` signify the largest world size
        till we need to duplicate. For instance if we have nheads=16 and
        world_size=32, then first 2 ranks will get first 1/16th of query
        """

        if self.fused:
            module_sharding_info = {
                "qkv_fused": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("qkv_fused"),
                    0,
                    [self.pre_tp_nheads, self.pre_tp_kvheads, self.pre_tp_kvheads],
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }
        else:
            module_sharding_info = {
                "query": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("query"), 0, [self.pre_tp_nheads]
                ),
                "key": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("key"), 0, [self.pre_tp_kvheads]
                ),
                "value": LinearModuleShardingInfo(
                    self.in_proj.get_submodule("value"), 0, [self.pre_tp_kvheads]
                ),
                "dense": LinearModuleShardingInfo(self.dense, 1, [self.world_size]),
            }

        type_sharding_map = get_all_linear_type_to_sharding_maps()
        unused_keys = type_sharding_map[self.linear_type](
            tensor_values,
            self,
            module_sharding_info,
        )
        return unused_keys

    @staticmethod
    def import_module(
        mha: MultiHeadAttention, group: ProcessGroup
    ) -> "TPMultiHeadAttention":
        tp_mha = TPMultiHeadAttention(
            emb_dim=mha.emb_dim,
            emb_kq=mha.emb_kq_per_head,
            emb_v=mha.emb_v_per_head,
            nheads=mha.nheads,
            kvheads=mha.kvheads,
            p_dropout=mha.p_dropout,
            use_bias=mha.use_bias,
            position_encoder=mha.position_encoder,
            group=group,
            fused=mha.fused,
            linear_config=mha.linear_config,
            scale_factor=mha.scale_factor,
        )
        return tp_mha

    def _copy_to_tp_region(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
    ):
        if (k is None and v is None) or (k is q and v is q):
            q_par = copy_to_tensor_model_parallel_region(q, self.group)
            if self.fused:
                k_par = None
                v_par = None
            else:
                k_par = copy_to_tensor_model_parallel_region(k, self.group)
                v_par = copy_to_tensor_model_parallel_region(v, self.group)
        else:
            raise ValueError(
                "both k and v must either be given as tensors or both None"
            )

        return q_par, k_par, v_par

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        position_ids=None,
        past_key_value_state: Optional[Tuple[Tensor | None, Tensor | None]] = None,
        use_cache=False,
        **attn_kwargs: Unpack[AttentionKwargs],
    ):
        """
        Check MultiHeadAttention for up-to-date arguments and docs
        """

        q_par, k_par, v_par = self._copy_to_tp_region(q, k, v)

        out_par = MultiHeadAttention.forward(
            self,
            q_par,
            k_par,
            v_par,
            position_ids,
            past_key_value_state,
            use_cache,
            **attn_kwargs,
        )

        # if use_cache=True, we return the hidden_state as well as the kv cache.
        # We only reduce the output, and keep the cache thread-local
        if use_cache:
            out = reduce_from_tensor_model_parallel_region(out_par[0], self.group)
            return out, out_par[1]
        else:
            out = reduce_from_tensor_model_parallel_region(out_par, self.group)
            return out
