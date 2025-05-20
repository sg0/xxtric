#include "deftypes.hpp"
#include "counting.cuh"
#include <cassert>
#include <iostream>

#if defined(USE_CUDA_CG)
#include <cooperative_groups.h> 
#include <cooperative_groups/memcpy_async.h>
namespace cg = cooperative_groups;
#if defined(USE_ASYNC_MEMCPY)
//#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
//static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_async_tx is not available.");
//#endif
/// https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async_tx.html#libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx
#include <cuda/barrier>
#include <cuda/std/utility>
using barrier = cuda::barrier<cuda::thread_scope_block>;
//namespace cde = cuda::device::experimental;
#endif
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
__device__ GraphElem lower_bound
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
      const GraphElem owner = lower_bound(size+1, edge_m, d_parts);

      if (g_i < edge_m)
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

              if (edge_n < d_range[base_edge_m*2])
                continue;

              atomicAdd((unsigned long long int*)(d_vcount+i), one);
              atomicAdd((unsigned long long int*)(d_send_count+owner), two);
            }
          }
        }
      }
    }
    
#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(FULL_MASK, count, j);
    
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
      const GraphElem owner = lower_bound(size+1, edge_m, d_parts); // get_owner

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

#if defined(LESS_GLOBALMEM_READS)
 void local_count_gpu
(
 const GraphElem lnv,
 const GraphElem base,
 const GraphElem bound,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
#else
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
#endif
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
#if defined(LESS_GLOBALMEM_READS)
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(lnv, base, bound, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_kernel<blocksize><<<nblocks, blocksize>>>(lnv, base, bound, d_indices, d_edge_list, d_local_count);
#endif
#else
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(lnv, base, bound, d_range, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_kernel<blocksize><<<nblocks, blocksize>>>(lnv, base, bound, d_range, d_indices, d_edge_list, d_local_count);
#endif
#endif
}

#if defined(USE_CUDA_CG)
#if defined(LESS_GLOBALMEM_READS)
template <int tilesize, int blocksize>
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
#else
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
#endif
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

#if defined(LESS_GLOBALMEM_READS)
#else
          if (edge_n > d_range[base_edge_m*2+1])
            break;

          if (edge_n < d_range[base_edge_m*2])
            continue;
#endif
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
    
#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(FULL_MASK, count, j);
    
    if (warp_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, count);
  }
}
#else
#if defined(LESS_GLOBALMEM_READS)
template <int blocksize>
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
#endif
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
        const GraphElem jstart = d_indices[base_edge_m];
        const GraphElem jend = d_indices[base_edge_m+1];
        
        for (GraphElem n = m + 1; n < e1; n++)
        {
          const GraphElem edge_n = d_edge_list[n];

#if defined(LESS_GLOBALMEM_READS)
#else
          if (edge_n > d_range[base_edge_m*2+1])
            break;

          if (edge_n < d_range[base_edge_m*2])
            continue;
#endif

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

/// single GPU counting
inline __device__ void count_if_edge_exists
(
 GraphElem const& e0, 
 GraphElem const& e1, 
 GraphElem const& se, 
 const GraphElem* __restrict__ d_edge_list,
 GraphElem& t_count
 )
{
  GraphElem const& first = d_edge_list[e0];
  GraphElem const& last = d_edge_list[e1-1];

  if (se == last || se == first)
  {
    t_count++;
    return;
  }

  if (se > first && se < last)
  {
    GraphElem const& mid = d_edge_list[MID(e0, e1)];

    if (se == mid)
    {
      t_count++;
      return;
    }
    else if (se > mid)
    {
      for (GraphElem e = MID(e0, e1) + 1; e < e1; e++)
      {
        if (d_edge_list[e] > se)
          break;
        if (d_edge_list[e] == se)
        {
          t_count++;
          return;
        }
      }
    }
    else
    {
      for (GraphElem e = e0; e < MID(e0, e1); e++)
      {
        if (d_edge_list[e] > se)
          break;
        if (d_edge_list[e] == se)
        {
          t_count++;
          return;
        }
      }
    }
  }
}

inline __device__ GraphElem next_power_of_2
(
 GraphElem x
)
{
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

inline __device__ GraphElem midpoint(GraphElem const& a, GraphElem const& b) { return (a + (b-a)/2); }

#if 0
inline __device__ void count_if_edge_exists_smem2
(
 GraphElem s1,
 GraphElem se, 
 volatile GraphElem* s_edge_list,
 GraphElem& t_count
 )
{
  const GraphElem s_first = s_edge_list[0];
  const GraphElem s_last  = s_edge_list[s1-1];

  if (se == s_first || se == s_last)
  {
    t_count += 1;
    return;
  }

  if (se > s_first && se < s_last)
  {
    GraphElem range = ((s1 & (s1 - 1)) == 0) ? s1 : next_power_of_2(s1); 
    GraphElem ret = 0;

#pragma unroll
    for (GraphElem s = range/2; s > 0; s /= 2)
      ret += (s_edge_list[ret + s] <= se) ? s : 0;

    t_count += (s_edge_list[ret] == se) ? 1 : 0;
  }
}
#endif

inline __device__ void count_if_edge_exists_smem2
(
 GraphElem s1,
 GraphElem se, 
 volatile GraphElem* s_edge_list,
 GraphElem& t_count
 )
{
  GraphElem beg = 0, end = s1, mid = midpoint(beg, end);
  const GraphElem s_first = s_edge_list[beg];
  const GraphElem s_last  = s_edge_list[end-1];

  if (se == s_first || se == s_last)
  {
    t_count += 1;
    return;
  }

  if (se > s_first && se < s_last)
  {
    while (end > beg) 
    { 
      if (s_edge_list[mid] == se)
      {
        t_count += 1;
        return;
      }
      else if (s_edge_list[mid] < se)
        beg = mid + 1;
      else
        end = mid;
    
      mid = midpoint(beg, end);
    }
  }
}

inline __device__ void count_if_edge_exists_smem
(
 GraphElem const& s1,
 GraphElem const& se, 
 volatile GraphElem* s_edge_list,
 GraphElem& t_count
 )
{
  const GraphElem s_first = s_edge_list[0];
  const GraphElem s_last  = s_edge_list[s1-1];
  const GraphElem mid = s_edge_list[midpoint(0, s1)];

  if (se == s_first || se == s_last || se == mid)
  {
    t_count++;
    return;
  }

  if (se > s_first && se < s_last)
  {
    if (se > mid)
    {
      for (GraphElem e = midpoint(0, s1) + 1; e < s1-1; e++)
      {
        if (s_edge_list[e] > se)
          return;
        if (s_edge_list[e] == se)
        {
          t_count++;
          return;
        }
      }
    }
    else
    {
      for (GraphElem e = 1; e < midpoint(0, s1); e++)
      {
        if (s_edge_list[e] > se)
          return;
        if (s_edge_list[e] == se)
        {
          t_count++;
          return;
        }
      }
    }
  }
}

#if 0
inline __device__ void count_if_edge_exists_smem
(
 GraphElem const& s1,
 GraphElem const& se, 
 volatile GraphElem* s_edge_list,
 GraphElem& t_count
 )
{
  const GraphElem s_first = s_edge_list[0];
  const GraphElem s_last  = s_edge_list[s1-1];

  if (se == s_first || se == s_last)
  {
    t_count++;
    return;
  }

  if (se > s_first && se < s_last)
  {
    GraphElem mid = s_edge_list[midpoint(0, s1)];

    if (se == mid)
    {
      t_count++;
      return;
    }
    else if (se > mid)
    {
      for (GraphElem e = midpoint(0, s1) + 1; e < s1-1; e++)
      {
        if (s_edge_list[e] > se)
          return;
        if (s_edge_list[e] == se)
        {
          t_count++;
          return;
        }
      }
    }
    else
    {
      for (GraphElem e = 1; e < midpoint(0, s1); e++)
      {
        if (s_edge_list[e] > se)
          return;
        if (s_edge_list[e] == se)
        {
          t_count++;
          return;
        }
      }
    }
  }
}
#endif

inline __device__ bool edge_exists
(
 GraphElem e0, 
 GraphElem e1, 
 GraphElem se, 
 const GraphElem* __restrict__ d_edge_list
 )
{
  GraphElem const& first = d_edge_list[e0];
  GraphElem const& last = d_edge_list[e1-1];

  if (se == last || se == first)
    return true;

  if (se > first && se < last)
  {
    GraphElem const& mid = d_edge_list[MID(e0, e1)];

    if (se == mid)
      return true;
    else if (se > mid)
    {
      for (GraphElem e = MID(e0, e1) + 1; e < e1; e++)
      {
        if (d_edge_list[e] > se)
          break;
        if (d_edge_list[e] == se)
          return true;
      }
    }
    else
    {
      for (GraphElem e = e0; e < MID(e0, e1); e++)
      {
        if (d_edge_list[e] > se)
          break;
        if (d_edge_list[e] == se)
          return true;
      }
    }
  }

  return false;
}

inline __device__ bool edge_exists_smem
(
 GraphElem s1,
 GraphElem se, 
 volatile GraphElem* s_edge_list
 )
{
  GraphElem s_first = s_edge_list[0];
  GraphElem s_last  = s_edge_list[s1-1];

  if (se == s_first || se == s_last)
    return true;

  if (se > s_first && se < s_last)
  {
    GraphElem mid = s_edge_list[MID(0, s1)];

    if (se == mid)
      return true;
    else if (se > mid)
    {
      for (GraphElem e = MID(0, s1) + 1; e < s1; e++)
      {
        if (s_edge_list[e] > se)
          break;
        if (s_edge_list[e] == se)
          return true;
      }
    }
    else
    {
      for (GraphElem e = 0; e < MID(0, s1); e++)
      {
        if (s_edge_list[e] > se)
          break;
        if (s_edge_list[e] == se)
          return true;
      }
    }
  }
  
  return false;
}

#if defined(SET_VERTEX_RANGE)
 void single_gpu_count
(
 const GraphElem nv,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_begin,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const GraphElem nblocks = (nv + blocksize - 1) / blocksize;

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
        single_gpu_count_kernel<TILESIZE01, blocksize>,
#else
        single_gpu_count_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);

    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Single GPU Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  

#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  single_gpu_count_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nv, d_indices, d_edge_begin, d_edge_list, d_local_count);
#else
  single_gpu_count_kernel<blocksize><<<nblocks, blocksize>>>(nv, d_indices, d_edge_begin, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
void single_gpu_count_kernel
(
 const GraphElem nv,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_begin,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  __shared__ GraphElem range[(BlockSize/WarpSize)*4];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*4];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

  for (GraphElem i = i0; i < nv; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid == 0)
    {
      const GraphElem id = 3*i;
      t_range[0] = d_edge_begin[id+1];
      t_range[1] = d_edge_begin[id+2];
      t_range[2] = d_indices[d_edge_begin[id]+1];
      t_range[3] = d_edge_begin[id];
    }

    warp.sync();

    const GraphElem start = t_range[0];
    const GraphElem end = t_range[1];
    const GraphElem last = t_range[2];
    const GraphElem vtx = t_range[3];

    for(GraphElem j = start + 1 + warp_tid; j < last; j += WarpSize)
    {
      const GraphElem edge_j = d_edge_list[j];

      for(GraphElem m = start; m <= end; m++)
      {
        const GraphElem edge_m = d_edge_list[m];
        
        if ((m != j) && (edge_j > edge_m))
        {
          const GraphElem nstart = d_indices[edge_m];
          const GraphElem nend = d_indices[edge_m+1];

          for(GraphElem n = nstart; n < nend; n++)
          {
            const GraphElem edge_n = d_edge_list[n];
            if (vtx > edge_n)
            {
              if (edge_n > edge_j)
                break;
              if (edge_n == edge_j)
              {
                count++;
                break;
              }
            }
          }
        }
      }
    }
  }
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);
  
  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#else
template <int blocksize>
__global__  
void single_gpu_count_kernel
(
 const GraphElem nv,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_begin,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
  __shared__ GraphElem range[4];
  sdata[tid] = 0;

  __syncthreads();

  for (GraphElem i = bid; i < nv; i += step)
  {
    if(tid == 0)
    {
      const GraphElem id = 3*i;
      range[0] = d_edge_begin[id+1];
      range[1] = d_edge_begin[id+2];
      range[2] = d_indices[d_edge_begin[id]+1];
      range[3] = d_edge_begin[id];
    }

    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];
    const GraphElem last = range[2];
    const GraphElem vtx = range[3];

    for(GraphElem j = e0 + 1 + tid; j < last; j += blocksize)
    {
      const GraphElem edge_j = d_edge_list[j];

      for(GraphElem m = e0; m <= e1; m++)
      {
        const GraphElem edge_m = d_edge_list[m];
        
        if ((m != j) && (edge_j > edge_m))
        {
          const GraphElem nstart = d_indices[edge_m];
          const GraphElem nend = d_indices[edge_m+1];

          for(GraphElem n = nstart; n < nend; n++)
          {
            const GraphElem edge_n = d_edge_list[n];
            if (vtx > edge_n)
            {
              if (edge_n > edge_j)
                break;
              if (edge_n == edge_j)
              {
                sdata[tid]++;
                break;
              }
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
#elif defined(SET_VERTEX_RANGE_SPLIT)
void single_gpu_count_masked
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const int tilesize = TILESIZE01;
  GraphElem nblocks = (nedges + (blocksize/tilesize) - 1) / (blocksize/tilesize);
  nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

#if defined(USE_CUDA_CG)
  single_gpu_count_masked_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
#else
  single_gpu_count_masked_kernel<blocksize><<<nblocks, blocksize>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
#endif
}

void single_gpu_count_masked
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
)
{
  const int blocksize = BLOCKDIM03;
  const int tilesize = TILESIZE01;
  GraphElem nblocks = (nedges + (blocksize/tilesize) - 1) / (blocksize/tilesize);
  nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;
#if defined(EXPAND_SM_CAPACITY)
  SHM_SZ = ;
  cudaFuncSetAttribute(
#if defined(USE_CUDA_CG)
      single_gpu_count_masked_kernel<tilesize, blocksize>,
#else
      single_gpu_count_masked_kernel<blocksize>,
#endif, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      SHM_SIZE);
#endif
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
        single_gpu_count_masked_kernel<tilesize, blocksize>,
#else
        single_gpu_count_masked_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);
    
    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Single GPU Masked Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
#if defined(USE_CUDA_CG)
  single_gpu_count_masked_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
#else
  single_gpu_count_masked_kernel<blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
#if 0
template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");

  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  const int nbrs_sm = 128;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];
  __shared__ GraphElem nbrs_j[(BlockSize/WarpSize)*nbrs_sm];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem* p_nbrs = &nbrs_j[(block_tid/WarpSize)*nbrs_sm];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

  for (GraphElem i = i0; i < nedges; i += step*(BlockSize/WarpSize))
  {
    GraphElem k;

    if(warp_tid == 0)
    {
      const GraphElem id = 2*i;
      t_range[0] = d_edge_mask[id];
      t_range[1] = d_edge_mask[id+1];
      k = 0;
    }

    warp.sync();

    const GraphElem e_end = t_range[0];
    const GraphElem m = t_range[1];

    const GraphElem edge_m = d_edge_list[m];
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

    if (warp_tid == 0)
    {
      for(GraphElem j = jstart; (j < jend) && (k < nbrs_sm); j++)
        p_nbrs[k++] = d_edge_list[j];
    }

    k = __shfl_sync(FULL_MASK, k, 0);

    for(GraphElem n = m + 1 + warp_tid; n < e_end; n += WarpSize)
    {
      if (((jstart+k) < jend) && edge_exists(jstart+k, jend, d_edge_list[n], d_edge_list))
        count++;
      if (edge_exists_smem(k, d_edge_list[n], p_nbrs))
        count++;
    }
  }

#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}

template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");

  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  const int nbrs_sm = 50;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];
  __shared__ GraphElem nbrs_j[(BlockSize/WarpSize)*nbrs_sm], nbrs_n[(BlockSize/WarpSize)*nbrs_sm];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem* j_nbrs = &nbrs_j[(block_tid/WarpSize)*nbrs_sm];
  GraphElem* n_nbrs = &nbrs_n[(block_tid/WarpSize)*nbrs_sm];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

  for (GraphElem i = i0; i < nedges; i += step*(BlockSize/WarpSize))
  {
    GraphElem t, nt;
    
    if(warp_tid == 0)
    {
      const GraphElem id = 2*i;
      t_range[0] = d_edge_mask[id];
      t_range[1] = d_edge_mask[id+1];
      t = 0;
      nt = 0;
    }

    warp.sync();

    const GraphElem e_end = t_range[0];
    const GraphElem m = t_range[1];

    const GraphElem edge_m = d_edge_list[m];
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

    if (warp_tid == 0)
    {
#pragma unroll
      for (GraphElem k = jstart, n = m + 1; (k < MIN(jstart+nbrs_sm, jend)) && (n < MIN(m+1+nbrs_sm, e_end)); k++, n++)
      { 
        j_nbrs[t++] = d_edge_list[k];
        n_nbrs[nt++] = d_edge_list[n];
      }
    }

    t = __shfl_sync(FULL_MASK, t, 0);
    nt = __shfl_sync(FULL_MASK, nt, 0);

    for(GraphElem n = m + 1 + warp_tid; n < e_end; n += WarpSize)
    {
      if (n < nt)
      {
        if (edge_exists_smem(t, n_nbrs[n-(m+1)], j_nbrs))
          count++;
      }
      else
      {
        if (((jstart+t) < jend) && (edge_exists(jstart+t, jend, d_edge_list[n], d_edge_list)))
          count++;
        if (edge_exists_smem(t, d_edge_list[n], j_nbrs))
          count++;
      }
    }
  }

#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}

template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");

  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize, SmemPerWarp = 128;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];
  __shared__ alignas(16) GraphElem nbrs[(BlockSize/WarpSize)*SmemPerWarp];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem* t_nbrs = &nbrs[(block_tid/WarpSize)*SmemPerWarp];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

#if defined(USE_ASYNC_MEMCPY)
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar[BlockSize/WarpSize];
  __shared__ barrier xbar[BlockSize/WarpSize];
  barrier* t_bar = &bar[block_tid/WarpSize];
  barrier* x_bar = &xbar[block_tid/WarpSize];
  if (warp_tid == 0)
  {
    init(t_bar, warp.size());
    init(x_bar, warp.size());
  }
  warp.sync();
#endif

  for (GraphElem i = i0; i < nedges; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid == 0)
    {
      const GraphElem id = 2*i;
      t_range[0] = d_edge_mask[id];
      t_range[1] = d_edge_mask[id+1];
    }

    warp.sync();

    const GraphElem e_end = t_range[0];
    const GraphElem m = t_range[1];

    const GraphElem edge_m = d_edge_list[m];
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

    for (GraphElem k = jstart; k < jend; k += SmemPerWarp)
    {
      const GraphElem smem_range = MIN(k + SmemPerWarp, jend);
      const GraphElem smem_count = smem_range - k;

#if defined(USE_ASYNC_MEMCPY)
        cuda::memcpy_async(warp, t_nbrs, &d_edge_list[k], cuda::aligned_size_t<sizeof(GraphElem)>(sizeof(GraphElem)*smem_count), *t_bar);
        //cuda::memcpy_async(warp, t_nbrs, &d_edge_list[k], sizeof(GraphElem)*smem_count, *t_bar);
#else
      if (warp_tid == 0)
      {
#if defined(USE_PIPELINED_MEMCPY)
        for (GraphElem t = k; t < smem_range; t++)
        {
          __pipeline_memcpy_async(&t_nbrs[t-k], &d_edge_list[t], sizeof(GraphElem));
          __pipeline_commit();
        }
        __pipeline_wait_prior(0);
#else
        for (GraphElem t = k; t < smem_range; t++)
          t_nbrs[t-k] = d_edge_list[t];
#endif
      }
#endif
      
#if defined(USE_ASYNC_MEMCPY)
      t_bar->arrive_and_wait();
#else
      warp.sync();
#endif

#pragma unroll
      for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists_smem(smem_count, d_edge_list[n], t_nbrs, count);
      
      warp.sync(); 
    }

    /*
#pragma unroll
      for (GraphElem t = k; t < smem_range; t++)
         t_nbrs[t-k] = d_edge_list[t];
      
      warp.sync();

#pragma unroll
      for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists_smem(smem_count, d_edge_list[n], t_nbrs, count);

      warp.sync();


#pragma unroll
    for (GraphElem k=jstart; k < MIN(jstart+SmemPerWarp, jend); k++) 
      t_nbrs[ns++] = d_edge_list[k];

    if (jstart + SmemPerWarp < jend)
    {
#pragma unroll
      for(GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists(jstart+ns, jend, d_edge_list[n], d_edge_list, count);
    }

#pragma unroll
    for(GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
      count_if_edge_exists_smem(ns, d_edge_list[n], t_nbrs, count);
   */
  }

#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#endif

template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 )
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");

  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize, SmemPerWarp = 750;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];
  __shared__ alignas(16) GraphElem nbrs[(BlockSize/WarpSize)*SmemPerWarp];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem* t_nbrs = &nbrs[(block_tid/WarpSize)*SmemPerWarp];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

#if defined(USE_ASYNC_MEMCPY)
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar[BlockSize/WarpSize];
  barrier* t_bar = &bar[block_tid/WarpSize];
  if (warp_tid == 0)
    init(t_bar, warp.size());
  warp.sync();
#endif

  for (GraphElem i = i0; i < nedges; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid == 0)
    {
      const GraphElem id = 2*i;
      t_range[0] = d_edge_mask[id];
      t_range[1] = d_edge_mask[id+1];
    }

    warp.sync();

    const GraphElem e_end = t_range[0];
    const GraphElem m = t_range[1];

    const GraphElem edge_m = d_edge_list[m];
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

#if defined(USE_ASYNC_MEMCPY)
    //barrier::arrival_token tok;
      //if (__any_sync(__activemask(), (d_edge_list[m+1] < d_edge_list[jstart]) || (d_edge_list[m+1] > d_edge_list[jend-1])))
      //  continue;
#endif

    for (GraphElem k = jstart; k < jend; k += SmemPerWarp)
    {
      const GraphElem smem_range = MIN(k + SmemPerWarp, jend);
      const GraphElem smem_count = smem_range - k;

#if defined(USE_ASYNC_MEMCPY)


      cuda::memcpy_async(warp, t_nbrs, &d_edge_list[k], cuda::aligned_size_t<sizeof(GraphElem)>(sizeof(GraphElem)*smem_count), *t_bar);
      /*
      if (warp_tid == 0)
      {  
        //cuda::memcpy_async(warp, t_nbrs, &d_edge_list[k], sizeof(GraphElem)*smem_count, *t_bar);
        cde::cp_async_bulk_global_to_shared(t_nbrs, &d_edge_list[k], cuda::aligned_size_t<sizeof(GraphElem)>(sizeof(GraphElem)*smem_count), *t_bar);
        tok = t_bar->arrive();
      } 
      else
        tok = t_bar->arrive();
        */
#else
      if (warp_tid == 0)
      {
#if defined(USE_PIPELINED_MEMCPY)
        for (GraphElem t = k; t < smem_range; t++)
        {
          __pipeline_memcpy_async(&t_nbrs[t-k], &d_edge_list[t], sizeof(GraphElem));
          __pipeline_commit();
        }
        __pipeline_wait_prior(0);
#else
        for (GraphElem t = k; t < smem_range; t++)
          t_nbrs[t-k] = d_edge_list[t];
#endif
      }
#endif
      
#if defined(USE_ASYNC_MEMCPY)
      t_bar->arrive_and_wait();
      //t_bar->wait(std::move(tok));
#else
      warp.sync();
#endif

#if defined(USE_ASYNC_MEMCPY)
#pragma unroll
      for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists_smem2(smem_count, d_edge_list[n], t_nbrs, count);
#else
#pragma unroll
      for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists_smem(smem_count, d_edge_list[n], t_nbrs, count);
#endif

      warp.sync(); 
    }
  }

#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}

#else
template <int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
  __shared__ GraphElem range[2];
  sdata[tid] = 0;
  volatile __shared__ GraphElem nbrs[blocksize];
  GraphElem ns = 0;

  __syncthreads();

  for (GraphElem i = bid; i < nedges; i += step)
  {
    if(tid == 0)
    {
      const GraphElem id = 2*i;
      range[0] = d_edge_mask[id];
      range[1] = d_edge_mask[id+1];
    }

    __syncthreads();

    const GraphElem e1 = range[0];
    const GraphElem start = range[1];
    const GraphElem edge_m = d_edge_list[start];
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

    for (GraphElem k = jstart; k < jend; k += blocksize) 
    {
      for (GraphElem t = k; t < MIN(k + blocksize, jend); t++)
        nbrs[ns++] = d_edge_list[t];
      
      __syncthreads();

      for(GraphElem n = start + 1 + tid; n < e1; n += blocksize)
      {
        if (edge_exists_smem(ns, d_edge_list[n], nbrs))
          sdata[tid]++;
      }
      
      __syncthreads();
      
      ns = 0;
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
#if 0
template <int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  volatile __shared__ GraphElem sdata[blocksize];
  const int scount = 256;
  __shared__ GraphElem range[2];
  sdata[tid] = 0;
  volatile __shared__ GraphElem nbrs[scount], n_nbrs[scount];

  __syncthreads();

  for (GraphElem i = bid; i < nedges; i += step)
  {
    GraphElem t, nt;
    
    if(tid == 0)
    {
      const GraphElem id = 2*i;
      range[0] = d_edge_mask[id];
      range[1] = d_edge_mask[id+1];
      t = 0; 
      nt = 0;
    }

    __syncthreads();

    const GraphElem e1 = range[0];
    const GraphElem start = range[1];
    const GraphElem edge_m = d_edge_list[start];
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];
    

    if (tid == 0)
    {
#pragma unroll
      for (GraphElem k=jstart; k < MIN(jstart+scount, jend); k++) 
        nbrs[t++] = d_edge_list[k];

#pragma unroll
      for (GraphElem n=start+1; n < MIN(start+1+scount, e1); n++) 
        n_nbrs[nt++] = d_edge_list[n];
    }

    t = __shfl_sync(FULL_MASK, t, 0);
    nt = __shfl_sync(FULL_MASK, nt, 0);
    //__syncthreads();

    for(GraphElem n = start+1+tid; n < e1; n += blocksize)
    {
      if (n < nt)
      {
        if (edge_exists_smem(t, n_nbrs[n-(start+1)], nbrs))
          sdata[tid]++;
      }
      else
      {
        if (edge_exists_smem(t, d_edge_list[n], nbrs))
          sdata[tid]++;
        if (((jstart+t) < jend) && edge_exists(jstart+t, jend, d_edge_list[n], d_edge_list))
          sdata[tid]++;
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
#endif
#else /// default single GPU kernel 
 void single_gpu_count
(
 const GraphElem nv,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
)
{
  const int blocksize = BLOCKDIM03;
  const int tilesize = TILESIZE01;
  GraphElem nblocks = (nv + (blocksize/tilesize) - 1) / (blocksize/tilesize);
  nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

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
        single_gpu_count_kernel<tilesize, blocksize>,
#else
        single_gpu_count_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);

    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Single GPU Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  

#if defined(USE_CUDA_CG)
  single_gpu_count_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nv, d_indices, d_edge_list, d_local_count);
#else
  single_gpu_count_kernel<blocksize><<<nblocks, blocksize>>>(nv, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
void single_gpu_count_kernel
(
 const GraphElem nv,
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
  GraphElem count = 0;

  for (GraphElem i = i0; i < nv; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid < 2)
      t_range[warp_tid] = d_indices[i+warp_tid];

    warp.sync();

    const GraphElem start = t_range[0];
    const GraphElem end = t_range[1];

    for(GraphElem m = start + warp_tid; m < end-1; m += WarpSize)
    {
      const GraphElem edge_m = d_edge_list[m];
      
      if (i > edge_m)
      {
        const GraphElem jstart = d_indices[edge_m];
        const GraphElem jend = d_indices[edge_m+1];

        for (GraphElem n = m + 1; n < end; n++)
        {
          const GraphElem edge_n = d_edge_list[n];

          if (i > edge_n)
          {
            if (edge_exists(jstart, jend, edge_n, d_edge_list))
              count++;
            /*
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
             */
          }
        }
      }
    }
  }
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);
  
  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#else
template <int blocksize>
__global__  
void single_gpu_count_kernel
(
 const GraphElem nv,
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

  for (GraphElem i = bid; i < nv; i += step)
  {
    if(tid < 2)
      range[tid] = d_indices[i+tid];
    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];

    for(GraphElem m = e0 + tid; m < e1-1; m += blocksize)
    {
      const GraphElem edge_m = d_edge_list[m];
      
      if (i > edge_m)
      {
        const GraphElem jstart = d_indices[edge_m];
        const GraphElem jend = d_indices[edge_m+1];

        for (GraphElem n = m + 1; n < e1; n++)
        {
          const GraphElem edge_n = d_edge_list[n];

          if (i > edge_n)
          {
            if (edge_exists(jstart, jend, edge_n, d_edge_list))
              sdata[tid]++;
            /*
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
            */
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
#if 0
template <int blocksize>
__global__  
void single_gpu_count_kernel
(
 const GraphElem nv,
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

  for (GraphElem i = bid; i < nv; i += step)
  {
    if(tid < 2)
      range[tid] = d_indices[i+tid];
    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];

    for(GraphElem m = e0 + tid; m < e1-1; m += blocksize)
    {
      const GraphElem edge_m = d_edge_list[m];
      
      if (i > edge_m)
      {
        const GraphElem jstart = d_indices[edge_m];
        const GraphElem jend = d_indices[edge_m+1];

        for (GraphElem n = m + 1; n < e1; n++)
        {
          const GraphElem edge_n = d_edge_list[n];

          if (i > edge_n)
          {
            if (edge_exists(jstart, jend, edge_n, d_edge_list))
              sdata[tid]++;
            /*
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
            */
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
#endif
#endif

void local_count_gpu_inner
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t* stream
 )
{
  const int blocksize = BLOCKDIM03;
  const int tilesize = TILESIZE01;
  GraphElem nblocks = ((end-start) + (blocksize/tilesize) - 1) / (blocksize/tilesize);
  nblocks = (nblocks > MAX_GRIDDIM) ? MAX_GRIDDIM : nblocks;

#if defined(USE_CUDA_CG)
  local_count_gpu_inner_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, (stream == nullptr? 0: *stream)>>>(start, end, jstart, jend, d_edge_list, d_local_count);
#else
  local_count_gpu_inner_kernel<blocksize><<<nblocks, blocksize, 0, (stream == nullptr? 0: *stream)>>>(start, end, jstart, jend, d_edge_list, d_local_count);
#endif
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

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
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
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  GraphElem count = 0;

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  const GraphElem b0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
    
  for (GraphElem i = start + b0; i < end; i += step*(BlockSize/WarpSize))
  {
    const GraphElem edge_i = d_edge_list[i];

    for(GraphElem k = jstart + warp_tid; k < jend; k += WarpSize)
    {
      GraphElem edge_k = d_edge_list[k];
      
      if (edge_k == edge_i)
        count++;

      if (__any_sync(__activemask(), edge_k >= edge_i))
        break;
      /*
      if (d_edge_list[k] > edge_i)
        break;
      if (d_edge_list[k] == edge_i)
      {
        count++;
        break;
      }
       */
    }
  }

 #pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#else
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
#endif

void local_count_gpu_ext
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem base,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list_local,
 const GraphElem* __restrict__ d_edge_list_other,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t* stream
 )
{
  const int blocksize = BLOCKDIM03;
  GraphElem nblocks = ((end-start) + blocksize - 1) / blocksize;

#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_ext_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, (stream == nullptr? 0: *stream)>>>(start, end, base, d_indices, d_edge_list_local, d_edge_list_other, d_local_count);
#else
  local_count_gpu_ext_kernel<blocksize><<<nblocks, blocksize, 0, (stream == nullptr? 0: *stream)>>>(start, end, base, d_indices, d_edge_list_local, d_edge_list_other, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__
  void local_count_gpu_ext_kernel
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem base,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list_local,
 const GraphElem* __restrict__ d_edge_list_other,
 GraphElem* __restrict__ d_local_count
 )
{
  static_assert(blocksize % tilesize == 0, "Block size must be a multiple of tile size");
  
  const GraphElem bid = blockIdx.x, step = gridDim.x;
  const int WarpSize = tilesize, BlockSize = blocksize;
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];
  GraphElem count = 0;

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  const GraphElem b0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
     
  if(warp_tid < 2) 
     range[warp_tid] = d_indices[base+warp_tid];

  warp.sync();
  
  const GraphElem jstart = range[0];
  const GraphElem jend = range[1];

  for (GraphElem i = start + b0; i < end; i += step*(BlockSize/WarpSize))
  {
    const GraphElem edge_i = d_edge_list_local[i];

    for(GraphElem k = jstart + warp_tid; k < jend; k += WarpSize)
    {
      if (d_edge_list_other[k] == edge_i)
        count++;

      if (__any_sync(__activemask(), d_edge_list_other[k] >= edge_i))
        break;
      /*
      if (d_edge_list[k] > edge_i)
        break;
      if (d_edge_list[k] == edge_i)
      {
        count++;
        break;
      }
       */
    }
  }

 #pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#else
template <int blocksize>
__global__
  void local_count_gpu_ext_kernel
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem base,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list_local,
 const GraphElem* __restrict__ d_edge_list_other,
 GraphElem* __restrict__ d_local_count
 )
{
  volatile __shared__ GraphElem sdata[blocksize];
  GraphElem tid = threadIdx.x, bid = blockIdx.x, step = gridDim.x;
  __shared__ GraphElem range[2];
  sdata[tid] = 0;

  if(tid < 2)
     range[tid] = d_indices[base+tid];
  __syncthreads();

  const GraphElem jstart = range[0];
  const GraphElem jend = range[1];

  for (GraphElem i = start + bid; i < end; i += step)
  {
    const GraphElem edge_i = d_edge_list_local[i];

    for(GraphElem k = jstart + tid; k < jend; k += blocksize)
    {
      if (d_edge_list_other[k] > edge_i)
        break;
      if (d_edge_list_other[k] == edge_i)
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

#if defined(USE_ASYNC_STREAMS_AT_RECV)
 void local_count_gpu_recv
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
)
{
  const int blocksize = BLOCKDIM02;
  GraphElem nblocks = (count + blocksize - 1) / blocksize;

#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_recv_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(count, base, d_rbuf, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_recv_kernel<blocksize><<<nblocks, blocksize, 0, stream>>>(count, base, d_rbuf, d_indices, d_edge_list, d_local_count);
#endif
}
#else
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
#endif

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
  GraphElem tid_count = 0;

  for (GraphElem i = c0; i < count/2; i += step*(BlockSize/WarpSize))
  {
    const GraphElem ind = 2*i;
    
    if(warp_tid < 2)
      t_range[warp_tid] = d_indices[d_rbuf[ind]-base+warp_tid];

    warp.sync();

    const GraphElem start = t_range[0];
    const GraphElem end = t_range[1];
    const GraphElem edge = d_rbuf[ind+1];

    for(GraphElem k = start + warp_tid; k < end; k += WarpSize)
    {
      if (d_edge_list[k] == edge)
        tid_count++;
      
      if (__any_sync(__activemask(), d_edge_list[k] >= edge))
        break;
    }
    /*
    for(GraphElem k = start + warp_tid; k < end; k += WarpSize)
    {
      if (d_edge_list[k] > edge)
        break;
      if (d_edge_list[k] == edge)
      {
        tid_count++;
        break;
      }
    }
    */
  }
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    tid_count += __shfl_down_sync(FULL_MASK, tid_count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, tid_count);
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
      if (d_edge_list[k] > edge)
        break;
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

#pragma unroll
    for(int j = warp.size()/2; j > 0; j /= 2)
      count += __shfl_down_sync(FULL_MASK, count, j);
    
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
  GraphElem count = 0;

  for (GraphElem i = i0; i < lnv; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid == 0)
    {
      const GraphElem id = 3*i;

      t_range[0] = d_edge_mask[id+1];
      t_range[1] = d_edge_mask[id+2];
      t_range[2] = d_indices[d_edge_mask[id]+1];
    }

    warp.sync();

    const GraphElem start = t_range[0];
    const GraphElem end = t_range[1];
    const GraphElem last = t_range[2];

    for(GraphElem j = start + 1 + warp_tid; j < last; j += WarpSize)
    {
      const GraphElem edge_j = d_edge_list[j];

      for(GraphElem m = start; m <= end; m++)
      {
        const GraphElem edge_m = d_edge_list[m];
        
        if ((m != j) && (edge_j > edge_m))
        {
          const GraphElem base_edge_m = edge_m-base;
          const GraphElem nstart = d_indices[base_edge_m];
          const GraphElem nend = d_indices[base_edge_m+1];

          for(GraphElem n = nstart; n < nend; n++)
          {
            const GraphElem edge_n = d_edge_list[n];

            if (edge_n > edge_j)
              break;
            if (edge_n == edge_j)
            {
              count++;
              break;
            }
          }
        }
      }
    }
  }
    
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
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
    if(tid == 0)
    {
      const GraphElem id = 3*i;
      range[0] = d_edge_mask[id+1];
      range[1] = d_edge_mask[id+2];
      range[2] = d_indices[d_edge_mask[id]+1];
    }

    __syncthreads();

    const GraphElem e0 = range[0];
    const GraphElem e1 = range[1];
    const GraphElem last = range[2];
   
    for(GraphElem j = e0 + 1 + tid; j < last; j += blocksize)
    {
      const GraphElem edge_j = d_edge_list[j];

      for(GraphElem m = e0; m <= e1; m++)
      {
        const GraphElem edge_m = d_edge_list[m];
        
        if ((m != j) && (edge_j > edge_m))
        {
          const GraphElem base_edge_m = edge_m-base;
          const GraphElem nstart = d_indices[base_edge_m];
          const GraphElem nend = d_indices[base_edge_m+1];

          for(GraphElem n = nstart; n < nend; n++)
          {
            const GraphElem edge_n = d_edge_list[n];

            if (edge_n > edge_j)
              break;
            if (edge_n == edge_j)
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

void local_count_gpu_masked3
(
 const GraphElem nedges,
 const GraphElem base,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
)
{
  const int blocksize = BLOCKDIM03;
  const GraphElem nblocks = (nedges + blocksize - 1) / blocksize;

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
        local_count_gpu_masked3_kernel<TILESIZE01, blocksize>,
#else
        local_count_gpu_masked3_kernel<blocksize>,
#endif
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);
    
    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Masked(3) Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
#if defined(USE_CUDA_CG)
  const int tilesize = TILESIZE01;
  local_count_gpu_masked3_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, base, d_edge_mask, d_indices, d_edge_list, d_local_count);
#else
  local_count_gpu_masked3_kernel<blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, base, d_edge_mask, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CUDA_CG)
template <int tilesize, int blocksize>
__global__  
 void local_count_gpu_masked3_kernel
(
 const GraphElem nedges,
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
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  GraphElem* t_range = &range[(block_tid/WarpSize)*2];
  GraphElem i0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

  for (GraphElem i = i0; i < nedges; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid == 0)
    {
      const GraphElem id = 2*i;
      t_range[0] = d_edge_mask[id];
      t_range[1] = d_edge_mask[id+1];
    }

    warp.sync();

    const GraphElem e_end = t_range[0];
    const GraphElem m = t_range[1];
    
    const GraphElem edge_m = d_edge_list[m];
    const GraphElem base_edge_m = edge_m-base;
    const GraphElem jstart = d_indices[base_edge_m];
    const GraphElem jend = d_indices[base_edge_m+1];

    for(GraphElem n = m + 1 + warp_tid; n < e_end; n += WarpSize)
    {
      const GraphElem edge_n = d_edge_list[n];
      
      for(GraphElem j = jstart; j < jend; j++)
      {
      /*
        if (d_edge_list[j] == edge_n)
          count++;

        if (__any_sync(__activemask(), d_edge_list[j] >= edge_n))
          break;
      */
        const GraphElem edge_j = d_edge_list[j];
        
        if (edge_j > edge_n)
          break;
        if (edge_j == edge_n)
        {
          count++;
          break;
        }
      }
    }
  }
    
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#else
template <int blocksize>
__global__  
 void local_count_gpu_masked3_kernel
(
 const GraphElem nedges,
 const GraphElem base,
 const GraphElem* __restrict__ d_edge_mask,
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

  for (GraphElem i = bid; i < nedges; i += step)
  {
    if(tid == 0)
    {
      const GraphElem id = 2*i;
      range[0] = d_edge_mask[id];
      range[1] = d_edge_mask[id+1];
    }

    __syncthreads();

    const GraphElem e1 = range[0];
    const GraphElem start = range[1];
    const GraphElem edge_m = d_edge_list[start];
    const GraphElem base_edge_m = edge_m-base;
    const GraphElem jstart = d_indices[base_edge_m];
    const GraphElem jend = d_indices[base_edge_m+1];

    for(GraphElem n = start + 1 + tid; n < e1; n += blocksize)
    {
      const GraphElem edge_n = d_edge_list[n];
      
      for(GraphElem j = jstart; j < jend; j++)
      {
        const GraphElem edge_j = d_edge_list[j];

        if (edge_j > edge_n)
          break;
        if (edge_j == edge_n)
        {
          sdata[tid]++;
          break;
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
