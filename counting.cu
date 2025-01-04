#include "deftypes.hpp"
#include "counting.cuh"
#include <cassert>

#if defined(USE_CUDA_CG) || defined(USE_CUDA_CG_REDUCE)
#include <cooperative_groups.h> 
namespace cg = cooperative_groups;
#endif

void local_count_gpu
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem bound,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem nblocks = (lnv + BLOCKDIM03 - 1) / BLOCKDIM03;
  local_count_gpu_kernel<<<nblocks, BLOCKDIM03>>>(lnv, base, bound, d_indices, d_edge_list, d_local_count);
}

__global__  
void local_count_gpu_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem bound,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, blocksize = blockDim.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[BLOCKDIM03];
  __shared__ GraphElem range[2];
  sdata[tid] = 0;
    
  __syncthreads();

  for (GraphElem i = bid; i < lnv; i += step)
  {
    const GraphElem g_i = i+base;
    if(tid < 2)
      range[tid] = d_indices[i+tid];
    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];

    for(GraphElem m = e0 + tid; m < e1; m += blocksize)
    {
      const GraphElem edge_m = d_edge_list[m];
      if (g_i < edge_m && (edge_m >= base && edge_m <= bound))
      {
        const GraphElem jstart = d_indices[edge_m-base];
        const GraphElem jend = d_indices[edge_m-base+1];
        for (GraphElem n = m + 1; n < e1; n++)
        {
          for(GraphElem j = jstart; j < jend; j++)
          {
            if (d_edge_list[n] == d_edge_list[j])
              sdata[tid]++;
          }
        }
      }
    }
  }

  __syncthreads();

  for (GraphElem s=blocksize/2; s>32; s>>=1) 
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32)
    unroll_last_warp(sdata, tid);   

  if (tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, sdata[0]);
}

void local_count_gpu_inner
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  GraphElem nblocks = ((end-start) + BLOCKDIM03 - 1) / BLOCKDIM03;
  local_count_gpu_inner_kernel<<<nblocks, BLOCKDIM03>>>(start, end, jstart, jend, d_edge_list, d_local_count);
}

__device__ void unroll_last_warp(volatile GraphElem *sdata, GraphElem tid)
{
  sdata[tid] += sdata[tid + 32];
  sdata[tid] += sdata[tid + 16];
  sdata[tid] += sdata[tid + 8];
  sdata[tid] += sdata[tid + 4];
  sdata[tid] += sdata[tid + 2];
  sdata[tid] += sdata[tid + 1];
}

__global__
  void local_count_gpu_inner_kernel
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  __shared__ GraphElem sdata[BLOCKDIM03];
  GraphElem tid = threadIdx.x;
  sdata[tid] = 0;
    
  __syncthreads();
  
  for (GraphElem i = start + blockIdx.x; i < end; i += gridDim.x)
  {
    for(GraphElem k = jstart + threadIdx.x; k < jend; k += blockDim.x)
    {
      if (d_edge_list[k] == d_edge_list[i])
        sdata[tid]++;
    }
  }
    
  __syncthreads();

  for (GraphElem s=blockDim.x/2; s>32; s>>=1) 
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32) 
    unroll_last_warp(sdata, tid);

  if (tid == 0) 
    atomicAdd((unsigned long long int*)d_local_count, sdata[0]);
}


 void local_count_gpu_recv
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  GraphElem nblocks = (count + BLOCKDIM03 - 1) / BLOCKDIM03;
  local_count_gpu_recv_kernel<<<nblocks, BLOCKDIM03>>>(count, base, d_rbuf,
      d_indices, d_edge_list, d_local_count);
}

#if defined(USE_CUDA_CG)
__global__
void local_count_gpu_recv_kernel
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  const GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = 32, BlockSize = BLOCKDIM03;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem c0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;

  for (GraphElem i = c0; i < count/2; i += step*(BlockSize/WarpSize))
  {
    const GraphElem ind = 2*i;
    
    if(warp_tid < 2)
      t_range[warp_tid] = d_indices[d_rbuf[ind]-base+warp_tid];

    warp.sync();

    const GraphElem start = t_range[0] + warp_tid;
    const GraphElem end = t_range[1];
    GraphElem count = 0;

    for(GraphElem k = start; k < end; k += WarpSize)
    {
      if (d_edge_list[k] == d_rbuf[ind+1])
        count++;
    }
    
    warp.sync();

    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(0xFFFFFFFF, count, j);
    
    warp.sync();

    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}
#else
__global__
void local_count_gpu_recv_kernel
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, blocksize = blockDim.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[BLOCKDIM03];
  __shared__ GraphElem range[2];
  sdata[tid] = 0;
    
  __syncthreads();
 
  for (GraphElem i = bid; i < count/2; i += step)
  {
    const GraphElem ind = 2*i;
    
    if(tid < 2)
      range[tid] = d_indices[d_rbuf[ind]-base+tid];
    __syncthreads();

    const GraphElem start = range[0];
    const GraphElem end = range[1];

    for(GraphElem k = start + tid; k < end; k += blocksize)
    {
      if (d_edge_list[k] == d_rbuf[ind+1])
        sdata[tid]++;
    }
  }

  __syncthreads();

  for (GraphElem s=blocksize/2; s>32; s>>=1) 
  {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid < 32)
    unroll_last_warp(sdata, tid);   

  if (tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, sdata[0]);
}
#endif

/// CPU kernels

// local counting
void local_count_cpu_inner
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 GraphElem* __restrict__ edge_list,
 GraphElem* d_local_count 
 )
{   
    for(GraphElem u = start; u < end; u++)
    {   
        for(GraphElem e = jstart; e < jend; e++)
        {
          if (edge_list[e] == edge_list[u])
            *d_local_count += 1;
        }
    }
}

void local_count_cpu_recv
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ rbuf,
 const GraphElem* __restrict__ indices,
 const GraphElem* __restrict__ edge_list,
 GraphElem* __restrict__ local_count
 )
{
  for (GraphElem i = 0; i < count; i += 2)
  {
    const GraphElem start = indices[rbuf[i]-base];
    const GraphElem end = indices[rbuf[i]-base+1];

    for(GraphElem k = start; k < end; k++)
    {
      if (edge_list[k] == rbuf[i+1])
        *local_count++;
    }
  }
}

