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
 const GraphElem length,
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 GraphElem* __restrict__ d_in_count
 )
{
  GraphElem nblocks = (length + BLOCKDIM03 - 1) / BLOCKDIM03;
  local_count_gpu_recv_kernel<<<nblocks, 2*BLOCKDIM03>>>(count, base, d_rbuf,
      d_indices, d_edge_list, d_local_count, d_in_count);
}

__global__
void local_count_gpu_recv_kernel
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 GraphElem* __restrict__ d_in_count
 )
{
  GraphElem tid = threadIdx.x;
  GraphElem index = blockDim.x*blockIdx.x+tid;
  GraphElem stride = blockDim.x*gridDim.x;

  __shared__ GraphElem sdata[2*BLOCKDIM03];
  sdata[tid] = 0;
  sdata[tid + BLOCKDIM03] = 0;
  GraphElem prev = 0;
  
  __syncthreads();
  
  for (GraphElem k = 0; k < count;)
  {
    if (d_rbuf[k] == -1)
    {
      k += 1;
      prev = k;
      continue;
    }

    GraphElem curr_count = 0;

    for (GraphElem m = k + 1; m < count; m++)
    {
      if (d_rbuf[m] == -1)
      {
        curr_count = m + 1;
        break;
      }

      const GraphElem start = d_indices[d_rbuf[k]-base];
      const GraphElem end = d_indices[d_rbuf[m]-base+1];

      for(GraphElem e = start+index; e < end; e += stride)
      {
        if (d_edge_list[e] == d_rbuf[m])
          sdata[tid]++;
      }
  
      sdata[tid + BLOCKDIM03]++;
    }

    k += (curr_count - prev);
    prev = k;
  }

  __syncthreads();

  for (GraphElem s=blockDim.x/2; s>32; s>>=1) 
  {
    if (tid < s)
    {
      sdata[tid] += sdata[tid + s];
      sdata[tid + BLOCKDIM03] += sdata[tid + BLOCKDIM03 + s];
    }
    __syncthreads();
  }

  if (tid < 32)
  {
    unroll_last_warp(sdata, tid);
    unroll_last_warp(sdata + BLOCKDIM03, tid);
  }

  if (tid == 0)
  { 
    *d_local_count += sdata[0];
    *d_in_count += sdata[BLOCKDIM03];
  }
}

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


