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
  
  for (GraphElem i = bid; i < count; i += step*2)
  {
    if(tid < 2)
      range[tid] = d_indices[d_rbuf[i]-base+tid];
    __syncthreads();

    const GraphElem start = range[0];
    const GraphElem end = range[1];
 
    for(GraphElem k = start + tid; k < end; k += blocksize)
    {
      if (d_edge_list[k] == d_rbuf[i+1])
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

/// CPU kernels

// local counting
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

