#include "deftypes.hpp"
#include "counting.cuh"
#include <cassert>
#include <iostream>

#if defined(USE_CUDA_CG)
#include <cooperative_groups.h> 
namespace cg = cooperative_groups;
#endif

#if defined(COMBINED_LOCAL_COUNTING_KERNEL)
void combined_local_count_gpu
(
 const GraphElem rank,
 const GraphElem size,
 const GraphElem lnv,
 const GraphElem* __restrict__ d_parts,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_vcount,
 GraphElem* __restrict__ d_send_count,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const GraphElem nblocks = (lnv + blocksize - 1) / blocksize;

#if defined(CALC_WARP_OCCUPANCY)
    int device, numBlocks;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    int len = 15;
    char gpu_id[15];

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
#if defined(USE_CUDA_CG)
        combined_local_count_gpu_kernel<TILESIZE01, blocksize>,
#else
        combined_local_count_gpu_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);

    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Combined Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  combined_local_count_gpu_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(rank, size, lnv, d_parts, d_range, d_indices, d_edge_list, d_vcount, d_send_count, d_local_count);
#else
  combined_local_count_gpu_kernel<blocksize><<<nblocks, blocksize>>>(rank, size, lnv, d_parts, d_range, d_indices, d_edge_list, d_vcount, d_send_count, d_local_count);
#endif
}

/// every thread launches
__device__ GraphElem get_owner
(
 const GraphElem size, 
 const GraphElem value, 
 const GraphElem* __restrict__ d_parts
)
{
  GraphElem lo = 0, hi = size-1;
  while(lo <= hi) 
  {
    GraphElem mid = lo + (hi - lo) / 2;
    if (d_parts[mid] <= value)
      lo = mid + 1;
    else 
      hi = mid - 1;
  }
  return lo-1;
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
void combined_local_count_gpu_kernel
(
 const GraphElem rank,
 const GraphElem size,
 const GraphElem lnv,
 const GraphElem* __restrict__ d_parts,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_vcount,
 GraphElem* __restrict__ d_send_count,
 GraphElem* __restrict__ d_local_count
)
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;

  const GraphElem base = d_parts[rank];
  const GraphElem one = 1, two = 2;

  warp.sync();

  for (GraphElem i = i0; i < lnv; i += step*(BlockSize/WarpSize))
  {
    const GraphElem g_i = i+base;
    
    if(warp_tid < 2)
      t_range[warp_tid] = d_indices[i+warp_tid];

    warp.sync();

    const GraphElem start = t_range[0] + warp_tid;
    const GraphElem end = t_range[1];
    GraphElem count = 0;

    for(GraphElem m = start; m < end; m += WarpSize)
    {
      const GraphElem edge_m = d_edge_list[m];
      const GraphElem base_edge_m = edge_m-base;
      const GraphElem owner = get_owner(size+1, edge_m, d_parts);

      if (g_i > edge_m)
      {
        if (owner == rank)
        {
          const GraphElem jstart = d_indices[base_edge_m];
          const GraphElem jend = d_indices[base_edge_m+1];

          for (GraphElem n = m + 1; n < end; n++)
          {
            const GraphElem edge_n = d_edge_list[n];

            if (edge_n > d_range[base_edge_m*2+1])
              break;

            if (edge_m > edge_n)
              continue;

            if (edge_n < d_range[base_edge_m*2])
              continue;

            for(GraphElem j = jstart; j < jend; j++)
            {
              if (d_edge_list[j] > edge_n)
                break;
              if (d_edge_list[j] == edge_n)
              {
                count++;
                break;
              }
            }
          }
        }
        else
        {
          if (owner > rank)
          {
            for (GraphElem n = m + 1; n < end; n++)
            {
              const GraphElem edge_n = d_edge_list[n];

              if (edge_n > d_range[base_edge_m*2+1])
                break;

              if (edge_m > edge_n)
                continue;

              if (edge_n < d_range[base_edge_m*2])
                continue;

              atomicAdd((unsigned long long int*)(d_vcount+i), one);
              atomicAdd((unsigned long long int*)(d_send_count+owner), two);
            }
          }
        }
      }
    }
    
    warp.sync();

#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(0xFFFFFFFF, count, j);
    
    warp.sync();

    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}

#else
template <int blocksize>
__global__  
void combined_local_count_gpu_kernel
(
 const GraphElem rank,
 const GraphElem size,
 const GraphElem lnv,
 const GraphElem* __restrict__ d_parts,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_vcount,
 GraphElem* __restrict__ d_send_count,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
  __shared__ GraphElem range[2];
  
  sdata[tid] = 0;
  const GraphElem base = d_parts[rank];
  const GraphElem one = 1, two = 2;
    
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
      const GraphElem base_edge_m = edge_m-base;
      const GraphElem owner = get_owner(size+1, edge_m, d_parts);

      if (g_i < edge_m)
      {
        if (owner == rank)
        {
          const GraphElem jstart = d_indices[edge_m-base];
          const GraphElem jend = d_indices[edge_m-base+1];

          for (GraphElem n = m + 1; n < e1; n++)
          {
            const GraphElem edge_n = d_edge_list[n];

            if (edge_n > d_range[base_edge_m*2+1])
              break;

            if (edge_m > edge_n)
              continue;

            if (edge_n < d_range[base_edge_m*2])
              continue;

            for(GraphElem j = jstart; j < jend; j++)
            {
              if (d_edge_list[j] > edge_n)
                break;
              if (d_edge_list[j] == edge_n)
              {
                sdata[tid]++;
                break;
              }
            }
          }
        }
        else
        {
          if (owner > rank)
          {
            for (GraphElem n = m + 1; n < e1; n++)
            {
              const GraphElem edge_n = d_edge_list[n];

              if (edge_n > d_range[base_edge_m*2+1])
                break;

              if (edge_m > edge_n)
                continue;

              if (edge_n < d_range[base_edge_m*2])
                continue;

              atomicAdd((unsigned long long int*)(d_vcount+i), one);
              atomicAdd((unsigned long long int*)(d_send_count+owner), two);
            }
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
#endif
#endif // combined_local_counting_kernel

void local_count_gpu
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem bound,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const GraphElem nblocks = (lnv + blocksize - 1) / blocksize;

#if defined(CALC_WARP_OCCUPANCY)
    int device, numBlocks;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    int len = 15;
    char gpu_id[15];

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
#if defined(USE_CUDA_CG)
        local_count_gpu_kernel<TILESIZE01, blocksize>,
#else
        local_count_gpu_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);

    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Separate Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(lnv, base, bound, d_range, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_kernel<blocksize><<<nblocks, blocksize>>>(lnv, base, bound, d_range, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
void local_count_gpu_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem bound,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;

  for (GraphElem i = i0; i < lnv; i += step*(BlockSize/WarpSize))
  {
    const GraphElem g_i = i+base;
    
    if(warp_tid < 2)
      t_range[warp_tid] = d_indices[i+warp_tid];

    warp.sync();

    const GraphElem start = t_range[0] + warp_tid;
    const GraphElem end = t_range[1];
    GraphElem count = 0;

    for(GraphElem m = start; m < end; m += WarpSize)
    {
      const GraphElem edge_m = d_edge_list[m];
      const GraphElem base_edge_m = edge_m-base;
      
      if (g_i < edge_m && (edge_m >= base && edge_m <= bound))
      {
        const GraphElem jstart = d_indices[base_edge_m];
        const GraphElem jend = d_indices[base_edge_m+1];
        
        for (GraphElem n = m + 1; n < end; n++)
        {
          const GraphElem edge_n = d_edge_list[n];

          if (edge_n > d_range[base_edge_m*2+1])
            break;

          if (edge_m > edge_n)
            continue;

          if (edge_n < d_range[base_edge_m*2])
            continue;

          for(GraphElem j = jstart; j < jend; j++)
          {
            if (d_edge_list[j] > edge_n)
              break;
            if (d_edge_list[j] == edge_n)
            {
              count++;
              break;
            }
          }
        }
      }
    }
    
    warp.sync();

#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(0xFFFFFFFF, count, j);
    
    warp.sync();

    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}
#else
template <int blocksize>
__global__  
void local_count_gpu_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem bound,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
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
      const GraphElem base_edge_m = edge_m-base;
      
      if (g_i < edge_m && (edge_m >= base && edge_m <= bound))
      {
        const GraphElem jstart = d_indices[edge_m-base];
        const GraphElem jend = d_indices[edge_m-base+1];
        
        for (GraphElem n = m + 1; n < e1; n++)
        {
          const GraphElem edge_n = d_edge_list[n];

          if (edge_n > d_range[base_edge_m*2+1])
            break;

          if (edge_m > edge_n)
            continue;

          if (edge_n < d_range[base_edge_m*2])
            continue;

          for(GraphElem j = jstart; j < jend; j++)
          {
            if (d_edge_list[j] > edge_n)
              break;
            if (d_edge_list[j] == edge_n)
            {
              sdata[tid]++;
              break;
            }
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
#endif

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
  const int blocksize = BLOCKDIM03;
  GraphElem nblocks = ((end-start) + blocksize - 1) / blocksize;
  local_count_gpu_inner_kernel<blocksize><<<nblocks, blocksize>>>(start, end, jstart, jend, d_edge_list, d_local_count);
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

template <int blocksize>
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
  volatile __shared__ GraphElem sdata[blocksize];
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  sdata[tid] = 0;
    
  __syncthreads();
  
  for (GraphElem i = start + bid; i < end; i += step)
  {
    const GraphElem edge_i = d_edge_list[i];
    for(GraphElem k = jstart + tid; k < jend; k += blocksize)
    {
      if (d_edge_list[k] > edge_i)
        break;
      if (d_edge_list[k] == edge_i)
      {
        sdata[tid]++;
        break;
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
  const int blocksize = BLOCKDIM02;
  GraphElem nblocks = (count + blocksize - 1) / blocksize;

#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_recv_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(count, base, d_rbuf,
      d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_recv_kernel<blocksize><<<nblocks, blocksize>>>(count, base, d_rbuf,
      d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
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
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
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
    const GraphElem edge = d_rbuf[ind+1];
    GraphElem count = 0;

    for(GraphElem k = start; k < end; k += WarpSize)
    {
      if (d_edge_list[k] == edge)
      {
        count++;
      }
    }
    
    warp.sync();

#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(0xFFFFFFFF, count, j);
    
    warp.sync();

    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}
#else
template <int blocksize>
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
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
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
    const GraphElem edge = d_rbuf[ind+1];

    for(GraphElem k = start + tid; k < end; k += blocksize)
    {
      if (d_edge_list[k] == edge)
      {
        sdata[tid]++;
        break;
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
#endif

void local_count_gpu_masked
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const GraphElem nblocks = (lnv + blocksize - 1) / blocksize;

#if defined(CALC_WARP_OCCUPANCY)
    int device, numBlocks;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    int len = 15;
    char gpu_id[15];


    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
#if defined(USE_CUDA_CG)
        local_count_gpu_masked_kernel<TILESIZE01, blocksize>,
#else
        local_count_gpu_masked_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);
    
    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Masked Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_masked_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(lnv, base, d_range, d_edge_mask, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_masked_kernel<blocksize><<<nblocks, blocksize>>>(lnv, base, d_range, d_edge_mask, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
 void local_count_gpu_masked_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  __shared__ GraphElem range[(BlockSize/WarpSize)*3];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*3];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;

  for (GraphElem i = i0; i < lnv; i += step*(BlockSize/WarpSize))
  {
    const GraphElem id = 3*i;
    
    if(warp_tid == 0)
    {
      t_range[0] = d_edge_mask[id+1];
      t_range[1] = d_edge_mask[id+2];
      t_range[2] = d_indices[d_edge_mask[id]+1];
    }

    warp.sync();

    const GraphElem start = t_range[0] + warp_tid;
    const GraphElem end = t_range[1];
    const GraphElem last = t_range[2];
    GraphElem count = 0;

    for(GraphElem m = start; m <= end; m += WarpSize)
    {
      const GraphElem edge_m = d_edge_list[m];
      const GraphElem base_edge_m = edge_m-base;
      const GraphElem jstart = d_indices[base_edge_m];
      const GraphElem jend = d_indices[base_edge_m+1];

      for (GraphElem n = m + 1; n < last; n++)
      {
        const GraphElem edge_n = d_edge_list[n];
                
        if (edge_n > d_range[base_edge_m*2+1])
          break;
                               
        if (edge_m > edge_n)
          continue;

        if (edge_n < d_range[base_edge_m*2])
          continue;
        
        for(GraphElem j = jstart; j < jend; j++)
        {
          if (d_edge_list[j] > edge_n)
            break;
          if (d_edge_list[j] == edge_n)
          {
            count++;
            break;
          }
        }
      }
    }
    
    warp.sync();

#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(0xFFFFFFFF, count, j);
    
    warp.sync();

    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}
#else
template <int blocksize>
__global__  
 void local_count_gpu_masked_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem* __restrict__ d_range,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
  __shared__ GraphElem range[3];
  sdata[tid] = 0;
    
  __syncthreads();

  for (GraphElem i = bid; i < lnv; i += step)
  {
    const GraphElem id = 3*i;

    if(tid == 0)
    {
      range[0] = d_edge_mask[id+1];
      range[1] = d_edge_mask[id+2];
      range[2] = d_indices[d_edge_mask[id]+1];
    }

    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];
    const GraphElem last = range[2];
    
    for(GraphElem m = e0 + tid; m <= e1; m += blocksize)
    {
      const GraphElem edge_m = d_edge_list[m];
      const GraphElem base_edge_m = edge_m-base;
      const GraphElem jstart = d_indices[base_edge_m];
      const GraphElem jend = d_indices[base_edge_m+1];
      
      for (GraphElem n = m + 1; n < last; n++)
      {
        const GraphElem edge_n = d_edge_list[n];
                
        if (edge_n > d_range[base_edge_m*2+1])
          break;
                               
        if (edge_m > edge_n)
          continue;

        if (edge_n < d_range[base_edge_m*2])
          continue;

        for(GraphElem j = jstart; j < jend; j++)
        {
          if (d_edge_list[j] > edge_n)
            break;
          if (d_edge_list[j] == edge_n)
          {
            sdata[tid]++;
            break;
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
#endif

/// masked/filtered kernels but with less global memory reads
void local_count_gpu_masked2
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const GraphElem nblocks = (lnv + blocksize - 1) / blocksize;

#if defined(CALC_WARP_OCCUPANCY)
    int device, numBlocks;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;
    int len = 15;
    char gpu_id[15];

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
#if defined(USE_CUDA_CG)
        local_count_gpu_masked2_kernel<TILESIZE01, blocksize>,
#else
        local_count_gpu_masked2_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);
    
    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Masked Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_masked2_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(lnv, base, d_edge_mask, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_masked2_kernel<blocksize><<<nblocks, blocksize>>>(lnv, base, d_edge_mask, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
 void local_count_gpu_masked2_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  __shared__ GraphElem range[(BlockSize/WarpSize)*3];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*3];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;

  for (GraphElem i = i0; i < lnv; i += step*(BlockSize/WarpSize))
  {
    const GraphElem id = 3*i;
    
    if(warp_tid == 0)
    {
      t_range[0] = d_edge_mask[id+1];
      t_range[1] = d_edge_mask[id+2];
      t_range[2] = d_indices[d_edge_mask[id]+1];
    }

    warp.sync();

    const GraphElem start = t_range[0] + warp_tid;
    const GraphElem end = t_range[1];
    const GraphElem last = t_range[2];
    GraphElem count = 0;

    for(GraphElem m = start; m <= end; m += WarpSize)
    {
      const GraphElem edge_m = d_edge_list[m];
      const GraphElem base_edge_m = edge_m-base;
      const GraphElem jstart = d_indices[base_edge_m];
      const GraphElem jend = d_indices[base_edge_m+1];

      for (GraphElem n = m + 1; n < last; n++)
      {
        const GraphElem edge_n = d_edge_list[n];
                
        for(GraphElem j = jstart; j < jend; j++)
        {
          if (d_edge_list[j] > edge_n)
            break;
          if (d_edge_list[j] == edge_n)
          {
            count++;
            break;
          }
        }
      }
    }
    
    warp.sync();

#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(0xFFFFFFFF, count, j);
    
    warp.sync();

    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}
#else
template <int blocksize>
__global__  
 void local_count_gpu_masked2_kernel
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
  __shared__ GraphElem range[3];
  sdata[tid] = 0;
    
  __syncthreads();

  for (GraphElem i = bid; i < lnv; i += step)
  {
    const GraphElem id = 3*i;

    if(tid == 0)
    {
      range[0] = d_edge_mask[id+1];
      range[1] = d_edge_mask[id+2];
      range[2] = d_indices[d_edge_mask[id]+1];
    }

    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];
    const GraphElem last = range[2];
    
    for(GraphElem m = e0 + tid; m <= e1; m += blocksize)
    {
      const GraphElem edge_m = d_edge_list[m];
      const GraphElem base_edge_m = edge_m-base;
      const GraphElem jstart = d_indices[base_edge_m];
      const GraphElem jend = d_indices[base_edge_m+1];
      
      for (GraphElem n = m + 1; n < last; n++)
      {
        const GraphElem edge_n = d_edge_list[n];
                
        for(GraphElem j = jstart; j < jend; j++)
        {
          if (d_edge_list[j] > edge_n)
            break;
          if (d_edge_list[j] == edge_n)
          {
            sdata[tid]++;
            break;
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
