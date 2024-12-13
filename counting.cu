#include "deftypes.hpp"
#include "counting.cuh"

#include <cooperative_groups.h> 
namespace cg = cooperative_groups;

void local_count_gpu
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
  local_count_gpu_kernel<<<nblocks, BLOCKDIM03>>>(start, end, jstart, jend, d_edge_list, d_local_count);
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
  void local_count_gpu_kernel
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


#if 0
 void local_count_gpu_recv
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 GraphElem* __restrict__ d_recv_buf,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  //GraphElem nblocks = (end - start) > MAX_GRIDDIM ? MAX_GRIDDIM : (end - start); 
  //local_count_gpu_kernel<<<nblocks, GRIDDIM>>>(start, end, jstart, jend, d_edge_list, d_local_count);
  GraphElem range = std::ceil((end - start) / 2)*2;
  if (range < BLOCKDIM04)
    range = BLOCKDIM04;
  GraphElem nblocks = (range / BLOCKDIM04) > MAX_GRIDDIM ? MAX_GRIDDIM : (range / BLOCKDIM04);
    
  local_count_gpu_recv_kernel<<<nblocks, BLOCKDIM04>>>(start, end, jstart, jend, d_edge_list, d_local_count);
}

__global__
  void local_count_gpu_recv_kernel
(
 const GraphElem count,
 const GraphElem nv,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 GraphElem* __restrict__ d_in_ghosts
 )
{
  __shared__ GraphElem sdata[BLOCKDIM03], tdata[BLOCKDIM03];
  GraphElem tid = threadIdx.x;
  sdata[tid] = 0;

  __shared__ GraphElem range[2];
  for(GraphElem i = blockIdx.x; i < count; i += gridDim.x)
  {
    if(threadIdx.x < 2)
      range[threadIdx.x] = d_indices[d_rbuf[i]-base+threadIdx.x];
    __syncthreads();

    GraphElem start = range[0];
    GraphElem end = range[1];

    for(GraphElem e = start+threadIdx.x; e < end; e += blockDim.x)
    {
      for (GraphElem k = i+1; k < count; k++)
      {
        if (d_edge_list[e] == d_rbuf[k])
          sdata[tid]++;
        tdata[tid]++;
      }
    }
  }

  __syncthreads();

  for (GraphElem s=blockDim.x/2; s>32; s>>=1) 
  {
    if (tid < s)
    {
      sdata[tid] += sdata[tid + s];
      tdata[tid] += tdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32)
  { 
    unroll_last_warp(sdata, tid);
    unroll_last_warp(tdata, tid);
  }

  if (tid == 0)
  { 
    atomicAdd((unsigned long long int*)d_local_count, sdata[0]);
    atomicAdd((unsigned long long int*)d_in_ghosts, tdata[0]);
  }
}
#endif

void local_count_cpu
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


