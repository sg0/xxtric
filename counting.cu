#include "deftypes.hpp"
#include "counting.cuh"
#include <cassert>
#include <iostream>

#include <cooperative_groups.h> 
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;
//#if defined(USE_ASYNC_MEMCPY)
//#if defined(__CUDA_MINIMUM_ARCH__) && __CUDA_MINIMUM_ARCH__ < 900
//static_assert(false, "Insufficient CUDA Compute Capability: cuda::device::memcpy_async_tx is not available.");
//#endif
/// https://nvidia.github.io/cccl/libcudacxx/extended_api/asynchronous_operations/memcpy_async_tx.html#libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx
#include <cuda/barrier>
#include <cuda/std/utility>
using barrier = cuda::barrier<cuda::thread_scope_block>;
//namespace cde = cuda::device::experimental;
//#endif

#include "tensor.cuh"

inline __device__ GraphElem lower_bound
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

inline __device__ bool edge_exists
(
 GraphElem e0, 
 GraphElem e1, 
 GraphElem se, 
 const GraphElem* __restrict__ d_edge_list
 )
{
  GraphElem beg = e0, end = e1, mid = midpoint(beg, end);
  const GraphElem first = d_edge_list[beg];
  const GraphElem last = d_edge_list[end-1];

  if (se == last || se == first)
    return true;

  if (se > first && se < last)
  {
    while (end > beg) 
    { 
      if (d_edge_list[mid] == se)
        return true;
      else if (d_edge_list[mid] < se)
        beg = mid + 1;
      else
        end = mid;

      mid = midpoint(beg, end);
    }
  }

  return false;
}

  
inline __device__ void count_if_edge_exists
(
 GraphElem e0, 
 GraphElem e1, 
 GraphElem se, 
 const GraphElem* __restrict__ d_edge_list,
 GraphElem& t_count
 )
{
  GraphElem beg = e0, end = e1, mid = midpoint(beg, end);
  const GraphElem first = d_edge_list[beg];
  const GraphElem last = d_edge_list[end-1];

  if (se == last || se == first)
  {
    t_count++;
    return;
  }

  if (se > first && se < last)
  {
    while (end > beg) 
    { 
      if (d_edge_list[mid] == se)
      {
        t_count += 1;
        return;
      }
      else if (d_edge_list[mid] < se)
        beg = mid + 1;
      else
        end = mid;

      mid = midpoint(beg, end);
    }
  }
}

inline __device__ void count_if_edge_exists_smem
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

inline __device__ void count_if_edge_exists_wmma
(
 GraphElem s1,
 GraphElem se, 
 volatile int* t_frag,
 GraphElem& t_count
 )
{
  GraphElem beg = 0, end = s1, mid = midpoint(beg, end);
  const GraphElem s_first = (GraphElem)t_frag[beg];
  const GraphElem s_last  = (GraphElem)t_frag[end-1];

  if (se == s_first || se == s_last)
  {
    t_count += 1;
    return;
  }

  if (se > s_first && se < s_last)
  {
    while (end > beg) 
    { 
      if ((GraphElem)(t_frag[mid]) == se)
      {
        t_count += 1;
        return;
      }
      else if ((GraphElem)(t_frag[mid]) < se)
        beg = mid + 1;
      else
        end = mid;
    
      mid = midpoint(beg, end);
    }
  }
}

 void single_gpu_count_wmma
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

  single_gpu_count_wmma_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
}

template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_wmma_kernel
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
  __shared__ GraphElem range[(BlockSize/WarpSize)*2];

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();
  
  const int tensorslots = 16;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag; 
  int* t_frag = &acc_frag.x[(block_tid/WarpSize)*tensorslots];

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
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

    for (GraphElem k = jstart; k < jend; k += tensorslots)
    {
      const GraphElem t_range = MIN(k + tensorslots, jend);
      const GraphElem t_count = t_range - k;

      for (GraphElem t = k; t < t_range; t++)
        t_frag[(int)(t-k)] = (int)(d_edge_list[t]);

#pragma unroll
      for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists_wmma(t_count, d_edge_list[n], t_frag, count);

      warp.sync(); 
    }
  }

#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}

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

#if defined(USE_CUDA_WMMA_PREFETCH)
  single_gpu_count_wmma_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
#else
  single_gpu_count_masked_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
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
        single_gpu_count_masked_kernel<tilesize, blocksize>,
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);
    
    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Single GPU Masked Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  
  single_gpu_count_masked_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, d_edge_mask, d_indices, d_edge_list, d_local_count);
}

#if defined(USE_ASYNC_MEMCPY)
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
  const int WarpSize = tilesize, BlockSize = blocksize, SmemPerWarp = 512;
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

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar[BlockSize/WarpSize];
  barrier* t_bar = &bar[block_tid/WarpSize];
  if (warp_tid == 0)
    init(t_bar, warp.size());
  warp.sync();

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

    //barrier::arrival_token tok;
      //if (__any_sync(__activemask(), (d_edge_list[m+1] < d_edge_list[jstart]) || (d_edge_list[m+1] > d_edge_list[jend-1])))
      //  continue;

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

#pragma unroll
      for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
        count_if_edge_exists_smem(smem_count, d_edge_list[n], t_nbrs, count);

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
    const GraphElem jstart = d_indices[edge_m];
    const GraphElem jend = d_indices[edge_m+1];

#pragma unroll
    for (GraphElem n = m+1+warp_tid; n < e_end; n += WarpSize)
      count_if_edge_exists(jstart, jend, d_edge_list[n], d_edge_list, count);
  }

#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#endif

/// default single GPU kernel 
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
        single_gpu_count_kernel<tilesize, blocksize>,
        blocksize,
        0);

    cudaDeviceGetPCIBusId(gpu_id, len, device);
    std::string str_gpu(gpu_id);

    activeWarps = numBlocks * blocksize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Single GPU Kernel Occupancy (" << str_gpu << "): " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
#endif  

  single_gpu_count_kernel<tilesize, blocksize><<<nblocks, blocksize>>>(nv, d_indices, d_edge_list, d_local_count);
}

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
          if (i > d_edge_list[n])
            count_if_edge_exists(jstart, jend, d_edge_list[n], d_edge_list, count);
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

  local_count_gpu_inner_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, (stream == nullptr? 0: *stream)>>>(start, end, jstart, jend, d_edge_list, d_local_count);
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

#if 0
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
    }
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

  cg::thread_block block = cg::this_thread_block();
  cg::thread_group warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  const GraphElem b0 = (block_tid/WarpSize)+(BlockSize/WarpSize)*bid;
  GraphElem count = 0;

  for (GraphElem i = start + b0; i < end; i += step*(BlockSize/WarpSize))
  {
    const GraphElem edge_i = d_edge_list[i];

    for(GraphElem k = jstart + warp_tid; k < jend; k += WarpSize)
    {
      if (__any_sync(__activemask(), edge_exists(k, MIN(k + WarpSize, jend), edge_i, d_edge_list)))
      {
        if (warp_tid == 0)
          count += 1;
        break;
      }
    }
  }

 #pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}

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
  const int tilesize = TILESIZE01;
  GraphElem nblocks = ((end-start) + blocksize - 1) / blocksize;

  local_count_gpu_ext_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, (stream == nullptr? 0: *stream)>>>(start, end, base, d_indices, d_edge_list_local, d_edge_list_other, d_local_count);
}

#if 0
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
      if (__any_sync(__activemask(), edge_exists(k, MIN(k + WarpSize, jend), edge_i, d_edge_list_other)))
      {
        if (warp_tid == 0)
          count += 1;
        break;
      }
    }
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
      if (__any_sync(__activemask(), edge_exists(k, MIN(k + WarpSize, jend), edge_i, d_edge_list_other)))
      {
        if (warp_tid == 0)
          atomicAdd((unsigned long long int*)d_local_count, 1);
        break;
      }
    }
  }
}

/// Distributed-memory kernels

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
  const int tilesize = TILESIZE01;
  GraphElem nblocks = (count + blocksize - 1) / blocksize;

  local_count_gpu_recv_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(count, base, d_rbuf, d_indices, d_edge_list, d_local_count);
}

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
  }
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    tid_count += __shfl_down_sync(FULL_MASK, tid_count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, tid_count);
}

void local_count_gpu_masked
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
  const int tilesize = TILESIZE01;
  GraphElem nblocks = (nedges + blocksize - 1) / blocksize;

#if defined(USE_CG_CLUSTER_GROUP)
  const int clustersize = std::max(TB_PER_CLUSTER,1);
  // pad grid dimension
  nblocks = ((nblocks + clustersize - 1) / clustersize) * clustersize;

  int device;
  cudaGetDevice(&device);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

  if (props.major >= 9) 
  {  
    // launch configuration
    cudaLaunchConfig_t config = {0};
    config.gridDim = nblocks;
    config.blockDim = blocksize;
    config.stream = stream;
    //config.dynamicSmemBytes = clustersize*sizeof(GraphElem);

    // cluster dimension attribute
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = clustersize; 
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    config.attrs = attribute;
    config.numAttrs = 1;

    cudaError_t err = 
      cudaLaunchKernelEx(
        &config,
        local_count_gpu_masked_cgcluster_kernel<tilesize, blocksize>, 
        nedges, 
        base, 
        d_edge_mask, 
        d_indices, 
        d_edge_list, 
        d_local_count
        );
    if (err != cudaSuccess)
      std::cerr << "CUDA Launch Error: " << cudaGetErrorString(err) << std::endl;
  }
  else
    std::cerr << "Insufficient CUDA Compute Capability: cg::cluster_group is not available." << std::endl;
#else
  local_count_gpu_masked_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, base, d_edge_mask, d_indices, d_edge_list, d_local_count);
#endif
}

#if defined(USE_CG_CLUSTER_GROUP)
template <int tilesize, int blocksize>
__global__ 
void local_count_gpu_masked_cgcluster_kernel
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

  cg::cluster_group cluster = cg::this_cluster();
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);

  const GraphElem cluster_tid = cluster.block_rank();
  const GraphElem block_tid = block.thread_rank();
  const GraphElem warp_tid = warp.thread_rank();

  const GraphElem WarpsPerBlock = BlockSize / WarpSize;
  const GraphElem WarpsPerGrid = step * WarpsPerBlock;

  const GraphElem warpID = block_tid / WarpSize;
  GraphElem* t_range = &range[warpID*2];
  GraphElem i0 =  warpID + (WarpsPerBlock * bid);
  GraphElem count = 0;

  for (GraphElem i = i0; i < nedges; i += WarpsPerGrid)
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
      count_if_edge_exists(jstart, jend, d_edge_list[n], d_edge_list, count);
  }
    
  // warp-level reduction
	__shared__ GraphElem block_smem[WarpsPerBlock]; 
  __shared__ GraphElem block_total_sum; 
  count = cg::reduce(warp, count, cg::plus<GraphElem>());
  
  if (warp_tid == 0)
    block_smem[warpID] = count;
  
  block.sync();

  // partial block-level reduction 
  if (warpID == 0)
  {
    GraphElem warp_result = (warp_tid < WarpsPerBlock) ? block_smem[warp_tid] : 0;
    GraphElem block_sum = cg::reduce(warp, warp_result, cg::plus<GraphElem>());

    if (warp_tid == 0)
      block_total_sum = block_sum; 
  }
 
	// cluster level reduction
  cluster.sync(); 
  GraphElem cluster_partial = 0;
  const GraphElem cluster_size = cluster.dim_blocks().x;

  if (cluster_tid == 0)
  { 
    if (warpID == 0) 
    {
      for (GraphElem r = warp_tid; r < cluster_size; r += WarpSize)
      {
        GraphElem* remote_smem = cluster.map_shared_rank(&block_total_sum, r);
        cluster_partial = *remote_smem;
      }

      GraphElem cluster_sum = cg::reduce(warp, cluster_partial, cg::plus<GraphElem>());

      if (warp_tid == 0)
        atomicAdd((unsigned long long int*)d_local_count, (unsigned long long int)cluster_sum);
    }
  }
  cluster.sync(); 
}
#else
template <int tilesize, int blocksize>
__global__  
 void local_count_gpu_masked_kernel
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
      count_if_edge_exists(jstart, jend, d_edge_list[n], d_edge_list, count);
  }
    
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}
#endif

void local_count_gpu_masked_ext
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices_other,
 const GraphElem* __restrict__ d_edge_list_other,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
)
{
  const int blocksize = BLOCKDIM03;
  const int tilesize = TILESIZE01;
  const GraphElem nblocks = (nedges + blocksize - 1) / blocksize;

  local_count_gpu_masked_ext_kernel<tilesize, blocksize><<<nblocks, blocksize, 0, stream>>>(nedges, d_edge_mask, d_indices_other, d_edge_list_other, d_edge_list, d_local_count);
}

template <int tilesize, int blocksize>
__global__ void local_count_gpu_masked_ext_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices_other,
 const GraphElem* __restrict__ d_edge_list_other,
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

  for (GraphElem i = i0; i < nedges; i += step*(BlockSize/WarpSize))
  {
    if(warp_tid == 0)
    {
      const GraphElem id = 3*i;
      t_range[0] = d_edge_mask[id];
      t_range[1] = d_edge_mask[id+1];
      t_range[2] = d_edge_mask[id+2];
    }

    warp.sync();

    const GraphElem e_end = t_range[0];
    const GraphElem m = t_range[1];
    const GraphElem base = t_range[2];
    
    const GraphElem jstart = d_indices_other[base];
    const GraphElem jend = d_indices_other[base+1];

    for(GraphElem n = m + 1 + warp_tid; n < e_end; n += WarpSize)
      count_if_edge_exists(jstart, jend, d_edge_list[n], d_edge_list_other, count);
  }
    
#pragma unroll
  for(int j = warp.size()/2; j > 0; j /= 2)
    count += __shfl_down_sync(FULL_MASK, count, j);

  if (warp_tid == 0)
    atomicAdd((unsigned long long int*)d_local_count, count);
}

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

#if defined(USE_CUDA_VMM_MPI)

#ifndef VMM_BLOCK
#define VMM_BLOCK (256)
#endif
#ifndef VMM_TILE
#define VMM_TILE (32)   // threads cooperating on one edge in the intra-node kernel
#endif

// ===========================================================================
// Device helpers
// ===========================================================================

/// first i in [lo, hi) with a[i] >= x (hi if none)
template <typename T>
__device__ __forceinline__ GraphElem dlower(const T* __restrict__ a, GraphElem lo, GraphElem hi, T x)
{
  while (lo < hi) {
    const GraphElem mid = lo + ((hi - lo) >> 1);
    if (a[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/// first i in [lo, hi) with a[i] > x (hi if none)
template <typename T>
__device__ __forceinline__ GraphElem dupper(const T* __restrict__ a, GraphElem lo, GraphElem hi, T x)
{
  while (lo < hi) {
    const GraphElem mid = lo + ((hi - lo) >> 1);
    if (a[mid] <= x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ bool dcontains(const GraphElem* __restrict__ a, GraphElem n, GraphElem x)
{
  if (n <= 0 || x < a[0] || x > a[n - 1]) return false;
  const GraphElem i = dlower(a, GraphElem(0), n, x);
  return i < n && a[i] == x;
}

/// GPU of this node owning global vertex v (v must be node-local).
__device__ __forceinline__ int owner_gpu(const NodeCSR& c, GraphElem v)
{
  // last h with vsplit[h] <= v
  return (int)(dupper(c.vsplit, GraphElem(0), (GraphElem)c.ngpus, v) - 1);
}

/// Adjacency of a node-local vertex v (may live on a peer GPU: VMM zero copy).
__device__ __forceinline__ const GraphElem* node_adj(const NodeCSR& c, GraphElem v, GraphElem& deg)
{
  const int h = owner_gpu(c, v);
  const GraphElem lv = v - c.vsplit[h];
  const GraphElem a0 = c.ind[h][lv], a1 = c.ind[h][lv + 1];
  deg = a1 - a0;
  return c.lst[h] + a0;
}

/// Source (local index) of local edge e of GPU g: ind[l] <= e < ind[l+1].
__device__ __forceinline__ GraphElem edge_source(const GraphElem* __restrict__ ind, GraphElem nv, GraphElem e)
{
  return dupper(ind, GraphElem(0), nv + 1, e) - 1;
}

static int sm_count()
{
  int dev = 0, n = 0;
  CudaCall(cudaGetDevice(&dev));
  CudaCall(cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev));
  return n > 0 ? n : 1;
}

static unsigned grid_for(GraphElem work, int per_block, int blocks_per_sm = 32)
{
  const GraphElem need = (work + per_block - 1) / per_block;
  const GraphElem cap  = (GraphElem)sm_count() * blocks_per_sm;
  return (unsigned)std::max<GraphElem>(1, std::min(need, cap));
}

// ===========================================================================
// erange (min/max neighbour) construction
// ===========================================================================

__global__ void k_build_erange(NodeCSR c, int g, GraphElem* __restrict__ erange)
{
  const GraphElem* ind = c.ind[g];
  const GraphElem* lst = c.lst[g];
  const GraphElem vb = c.vsplit[g], nv = c.vsplit[g + 1] - vb;
  for (GraphElem l = blockIdx.x * (GraphElem)blockDim.x + threadIdx.x; l < nv;
       l += (GraphElem)gridDim.x * blockDim.x) {
    const GraphElem e0 = ind[l], e1 = ind[l + 1];
    if (e1 - e0 < 2) continue;  // cannot close a triangle; keep the empty [0,0]
    const GraphElem u = vb + l;
    erange[2 * u]     = lst[e0];
    erange[2 * u + 1] = lst[e1 - 1];
  }
}

void vmm_build_erange(const NodeCSR& csr, int g, GraphElem nv, GraphElem* erange,
                      cudaStream_t stream)
{
  if (nv <= 0) return;
  k_build_erange<<<grid_for(nv, VMM_BLOCK), VMM_BLOCK, 0, stream>>>(csr, g, erange);
  CudaCall(cudaGetLastError());
}

// ===========================================================================
// Intra-node counting (replaces local + IPC-neighbour kernels and their
// host-built edge masks). One tile per edge (u,v) of GPU g:
//   skip unless u < v and v is on this node;
//   adj(v) is read wherever it lives on the node (VMM mapping);
//   w ranges over adj(u) after v, pruned to [min adj(v), max adj(v)].
// ===========================================================================

template <int TILE>
__global__ void k_intranode(NodeCSR c, int g, GraphElem ne, VmmCount* __restrict__ count)
{
  auto tile = cg::tiled_partition<TILE>(cg::this_thread_block());
  const GraphElem* ind = c.ind[g];
  const GraphElem* lst = c.lst[g];
  const GraphElem vb = c.vsplit[g], nv = c.vsplit[g + 1] - vb;
  const GraphElem nbase = c.vsplit[0], nbound = c.vsplit[c.ngpus];

  const GraphElem ntiles = ((GraphElem)gridDim.x * blockDim.x) / TILE;
  GraphElem t = (blockIdx.x * (GraphElem)blockDim.x + threadIdx.x) / TILE;

  VmmCount my = 0;
  for (GraphElem e = t; e < ne; e += ntiles) {
    GraphElem nlo = 0, nhi = 0, deg = 0;
    unsigned long long adjp = 0;
    if (tile.thread_rank() == 0) {
      const GraphElem l = edge_source(ind, nv, e);
      const GraphElem u = vb + l, v = lst[e], eend = ind[l + 1];
      if (u < v && v >= nbase && v < nbound && e + 1 < eend) {
        const GraphElem* adj = node_adj(c, v, deg);
        if (deg >= 2) {
          nlo  = dlower(lst, e + 1, eend, adj[0]);
          nhi  = dupper(lst, nlo, eend, adj[deg - 1]);
          adjp = reinterpret_cast<unsigned long long>(adj);
        }
      }
    }
    nlo  = tile.shfl(nlo, 0);
    nhi  = tile.shfl(nhi, 0);
    deg  = tile.shfl(deg, 0);
    adjp = tile.shfl(adjp, 0);
    const GraphElem* adj = reinterpret_cast<const GraphElem*>(adjp);
    for (GraphElem n = nlo + tile.thread_rank(); n < nhi; n += TILE)
      if (dcontains(adj, deg, lst[n])) ++my;
  }
  my = cg::reduce(tile, my, cg::plus<VmmCount>());
  if (tile.thread_rank() == 0 && my) atomicAdd(count, my);
}

void vmm_count_intranode(const NodeCSR& csr, int g, GraphElem ne, VmmCount* d_count,
                         cudaStream_t stream)
{
  if (ne <= 0) return;
  const unsigned grid = grid_for(ne, VMM_BLOCK / VMM_TILE);
  k_intranode<VMM_TILE><<<grid, VMM_BLOCK, 0, stream>>>(csr, g, ne, d_count);
  CudaCall(cudaGetLastError());
}

// ===========================================================================
// Remote plan (GPU-side replacement of the host send-side buffering)
// ===========================================================================

/// Pair run of local edge e of GPU g (0 if e is not a remote item).
/// Returns destination node via `dest`, first n via `nlo`.
__device__ __forceinline__ GraphElem remote_run(const NodeCSR& c, int g, GraphElem e,
                                                const GraphElem* __restrict__ parts, int nparts,
                                                const GraphElem* __restrict__ erange,
                                                int& dest, GraphElem& nlo)
{
  const GraphElem* ind = c.ind[g];
  const GraphElem* lst = c.lst[g];
  const GraphElem vb = c.vsplit[g], nv = c.vsplit[g + 1] - vb;
  const GraphElem l = edge_source(ind, nv, e);
  const GraphElem u = vb + l, v = lst[e], eend = ind[l + 1];

  if (!(u < v) || (v >= c.vsplit[0] && v < c.vsplit[c.ngpus]) || e + 1 >= eend)
    return 0;

  GraphElem lo = e + 1, hi = eend;
  if (erange) {  // prune w to [min adj(v), max adj(v)] (as edge_above_min/within_max)
    const GraphElem emin = erange[2 * v], emax = erange[2 * v + 1];
    lo = dlower(lst, lo, hi, emin);
    hi = dupper(lst, lo, hi, emax);
  }
  if (hi <= lo) return 0;
  dest = (int)(dupper(parts, GraphElem(0), (GraphElem)nparts + 1, v) - 1);
  nlo  = lo;
  return hi - lo;
}

/// WRITE=false: count remote items. WRITE=true: compact (dest, edge) of the
/// remote items with warp-aggregated atomics (order is irrelevant: we sort).
template <bool WRITE>
__global__ void k_classify(NodeCSR c, int g, GraphElem ne, const GraphElem* __restrict__ parts,
                           int nparts, const GraphElem* __restrict__ erange,
                           unsigned long long* __restrict__ counter,
                           int* __restrict__ keys, GraphElem* __restrict__ items)
{
  const unsigned lane = threadIdx.x & 31u;
  const GraphElem stride = (GraphElem)gridDim.x * blockDim.x;
  // Warp-uniform loop so that __ballot_sync sees the full warp.
  for (GraphElem base = blockIdx.x * (GraphElem)blockDim.x + (threadIdx.x & ~31u); base < ne;
       base += stride) {
    const GraphElem e = base + lane;
    int dest = -1;
    GraphElem nlo = 0, cnt = 0;
    if (e < ne) cnt = remote_run(c, g, e, parts, nparts, erange, dest, nlo);
    const unsigned mask = __ballot_sync(0xffffffffu, cnt > 0);
    if (!mask) continue;
    unsigned long long off = 0;
    if (lane == 0) off = atomicAdd(counter, (unsigned long long)__popc(mask));
    if (WRITE) {
      off = __shfl_sync(0xffffffffu, off, 0);
      if (cnt > 0) {
        const unsigned long long pos = off + __popc(mask & ((1u << lane) - 1u));
        keys[pos]  = dest;
        items[pos] = e;
      }
    }
  }
}

__global__ void k_plan_gather(NodeCSR c, int g, GraphElem nrem, const GraphElem* __restrict__ parts,
                              int nparts, const GraphElem* __restrict__ erange,
                              const GraphElem* __restrict__ items, GraphElem* __restrict__ nlo,
                              GraphElem* __restrict__ cnt)
{
  for (GraphElem j = blockIdx.x * (GraphElem)blockDim.x + threadIdx.x; j <= nrem;
       j += (GraphElem)gridDim.x * blockDim.x) {
    if (j == nrem) { cnt[j] = 0; continue; }  // extra slot -> posx[nrem] = total
    int dest; GraphElem lo = 0;
    cnt[j] = remote_run(c, g, items[j], parts, nparts, erange, dest, lo);
    nlo[j] = lo;
  }
}

__global__ void k_plan_dests(GraphElem nrem, int nparts, const int* __restrict__ keys,
                             const GraphElem* __restrict__ posx, GraphElem* __restrict__ dbeg,
                             GraphElem* __restrict__ doff, GraphElem* __restrict__ dtot)
{
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < nparts;
       s += gridDim.x * blockDim.x) {
    const GraphElem b = dlower(keys, GraphElem(0), nrem, s);
    const GraphElem e = dlower(keys, b, nrem, s + 1);
    dbeg[s] = b;
    doff[s] = posx[b];
    dtot[s] = posx[e] - posx[b];
    if (s == nparts - 1) dbeg[nparts] = nrem;
  }
}

static int bits_for(int n)
{
  int b = 1;
  while ((1LL << b) <= n) ++b;
  return b;
}

void vmm_plan_remote(const NodeCSR& csr, int g, GraphElem ne, const GraphElem* d_parts,
                     int nparts, int /*my_part*/, const GraphElem* erange,
                     VmmRemotePlan& plan, cudaStream_t stream)
{
  plan = VmmRemotePlan{};
  CudaCall(cudaMallocAsync(&plan.dbeg, (nparts + 1) * sizeof(GraphElem), stream));
  CudaCall(cudaMallocAsync(&plan.doff, nparts * sizeof(GraphElem), stream));
  CudaCall(cudaMallocAsync(&plan.dtot, nparts * sizeof(GraphElem), stream));
  CudaCall(cudaMemsetAsync(plan.dbeg, 0, (nparts + 1) * sizeof(GraphElem), stream));
  CudaCall(cudaMemsetAsync(plan.doff, 0, nparts * sizeof(GraphElem), stream));
  CudaCall(cudaMemsetAsync(plan.dtot, 0, nparts * sizeof(GraphElem), stream));
  if (ne <= 0) { CudaCall(cudaStreamSynchronize(stream)); return; }

  // 1. count remote items
  unsigned long long* d_counter = nullptr;
  CudaCall(cudaMallocAsync(&d_counter, sizeof(unsigned long long), stream));
  CudaCall(cudaMemsetAsync(d_counter, 0, sizeof(unsigned long long), stream));
  const unsigned grid = grid_for(ne, VMM_BLOCK);
  k_classify<false><<<grid, VMM_BLOCK, 0, stream>>>(csr, g, ne, d_parts, nparts, erange,
                                                     d_counter, nullptr, nullptr);
  CudaCall(cudaGetLastError());
  unsigned long long nrem = 0;
  CudaCall(cudaMemcpyAsync(&nrem, d_counter, sizeof(nrem), cudaMemcpyDeviceToHost, stream));
  CudaCall(cudaStreamSynchronize(stream));
  if (nrem == 0) { CudaCall(cudaFreeAsync(d_counter, stream)); CudaCall(cudaStreamSynchronize(stream)); return; }
  if (nrem >= (unsigned long long)INT_MAX) {
    std::fprintf(stderr, "vmm_plan_remote: %llu remote items on one GPU exceed the CUB int limit\n", nrem);
    std::abort();
  }
  plan.nrem = (GraphElem)nrem;
  const int n = (int)nrem;

  // 2. compact (dest, edge)
  int* keys = nullptr; int* keys_s = nullptr; GraphElem* items = nullptr; GraphElem* cnt = nullptr;
  CudaCall(cudaMallocAsync(&keys,   n * sizeof(int), stream));
  CudaCall(cudaMallocAsync(&keys_s, n * sizeof(int), stream));
  CudaCall(cudaMallocAsync(&items,  n * sizeof(GraphElem), stream));
  CudaCall(cudaMallocAsync(&plan.items, n * sizeof(GraphElem), stream));
  CudaCall(cudaMemsetAsync(d_counter, 0, sizeof(unsigned long long), stream));
  k_classify<true><<<grid, VMM_BLOCK, 0, stream>>>(csr, g, ne, d_parts, nparts, erange,
                                                    d_counter, keys, items);
  CudaCall(cudaGetLastError());

  // 3. group by destination (radix sort on the low bits only)
  void* tmp = nullptr; size_t tmp_bytes = 0, need = 0;
  CudaCall(cub::DeviceRadixSort::SortPairs(nullptr, need, keys, keys_s, items, plan.items, n,
                                             0, bits_for(nparts), stream));
  tmp_bytes = need;
  CudaCall(cudaMallocAsync(&tmp, tmp_bytes, stream));
  CudaCall(cub::DeviceRadixSort::SortPairs(tmp, tmp_bytes, keys, keys_s, items, plan.items, n,
                                             0, bits_for(nparts), stream));
  CudaCall(cudaFreeAsync(keys, stream));
  CudaCall(cudaFreeAsync(items, stream));

  // 4. per-item run start + count, exclusive scan -> pair positions
  CudaCall(cudaMallocAsync(&plan.nlo,  n * sizeof(GraphElem), stream));
  CudaCall(cudaMallocAsync(&cnt,  (n + 1) * sizeof(GraphElem), stream));
  CudaCall(cudaMallocAsync(&plan.posx, (n + 1) * sizeof(GraphElem), stream));
  k_plan_gather<<<grid_for(n + 1, VMM_BLOCK), VMM_BLOCK, 0, stream>>>(
      csr, g, plan.nrem, d_parts, nparts, erange, plan.items, plan.nlo, cnt);
  CudaCall(cudaGetLastError());

  need = 0;
  CudaCall(cub::DeviceScan::ExclusiveSum(nullptr, need, cnt, plan.posx, n + 1, stream));
  if (need > tmp_bytes) {
    CudaCall(cudaFreeAsync(tmp, stream));
    tmp_bytes = need;
    CudaCall(cudaMallocAsync(&tmp, tmp_bytes, stream));
  }
  CudaCall(cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, cnt, plan.posx, n + 1, stream));

  // 5. destination ranges
  k_plan_dests<<<grid_for(nparts, VMM_BLOCK), VMM_BLOCK, 0, stream>>>(
      plan.nrem, nparts, keys_s, plan.posx, plan.dbeg, plan.doff, plan.dtot);
  CudaCall(cudaGetLastError());

  CudaCall(cudaMemcpyAsync(&plan.npairs, plan.posx + n, sizeof(GraphElem),
                             cudaMemcpyDeviceToHost, stream));
  CudaCall(cudaFreeAsync(tmp, stream));
  CudaCall(cudaFreeAsync(cnt, stream));
  CudaCall(cudaFreeAsync(keys_s, stream));
  CudaCall(cudaFreeAsync(d_counter, stream));
  CudaCall(cudaStreamSynchronize(stream));
}

void vmm_plan_free(VmmRemotePlan& plan)
{
  // cudaFree on nullptr is a no-op. Caller has the owning device current.
  cudaFree(plan.items); cudaFree(plan.nlo); cudaFree(plan.posx);
  cudaFree(plan.dbeg);  cudaFree(plan.doff); cudaFree(plan.dtot);
  plan = VmmRemotePlan{};
}

// ===========================================================================
// Pair generation: GPU fills its own send buffers for round k
// (replaces lookup_edges() + sbuf_/sbuf_ctr_/prev_m_/prev_k_ on the host)
// ===========================================================================

__global__ void k_generate(NodeCSR c, int g, const GraphElem* __restrict__ items,
                           const GraphElem* __restrict__ nlo, const GraphElem* __restrict__ posx,
                           const VmmSendSlot* __restrict__ slots, int nslots, int round,
                           GraphElem* __restrict__ sbuf)
{
  const GraphElem* lst = c.lst[g];
  for (int s = blockIdx.y; s < nslots; s += gridDim.y) {
    const VmmSendSlot S = slots[s];
    const GraphElem start = (GraphElem)round * S.cap;
    if (start >= S.tot) continue;
    const GraphElem len = (S.tot - start < S.cap) ? (S.tot - start) : S.cap;
    GraphElem* out = sbuf + S.bufoff + (GraphElem)(round % S.nsets) * S.cap * 2;

    for (GraphElem p = blockIdx.x * (GraphElem)blockDim.x + threadIdx.x; p < len;
         p += (GraphElem)gridDim.x * blockDim.x) {
      const GraphElem gp = S.doff + start + p;
      // item j with posx[j] <= gp < posx[j+1]; every item in [dbeg,dend) has cnt > 0
      const GraphElem j = dupper(posx, S.dbeg, S.dend, gp) - 1;
      out[2 * p]     = lst[items[j]];                 // v (owned by S.dest)
      out[2 * p + 1] = lst[nlo[j] + (gp - posx[j])];  // w
    }
  }
}

void vmm_generate_pairs(const NodeCSR& csr, int g, const VmmRemotePlan& plan,
                        const VmmSendSlot* d_slots, int nslots, GraphElem max_cap, int round,
                        GraphElem* sbuf, cudaStream_t stream)
{
  if (nslots <= 0 || max_cap <= 0) return;
  dim3 grid(grid_for(max_cap, VMM_BLOCK, 8), (unsigned)std::min(nslots, 65535));
  k_generate<<<grid, VMM_BLOCK, 0, stream>>>(csr, g, plan.items, plan.nlo, plan.posx, d_slots,
                                             nslots, round, sbuf);
  CudaCall(cudaGetLastError());
}

// ===========================================================================
// Receive-side counting directly on the (RDMA-written) VMM receive buffer
// ===========================================================================

__global__ void k_count_recv(NodeCSR c, const GraphElem* __restrict__ rbuf, GraphElem npairs,
                             VmmCount* __restrict__ count)
{
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  VmmCount my = 0;
  for (GraphElem i = blockIdx.x * (GraphElem)blockDim.x + threadIdx.x; i < npairs;
       i += (GraphElem)gridDim.x * blockDim.x) {
    const GraphElem v = rbuf[2 * i], w = rbuf[2 * i + 1];
    GraphElem deg = 0;
    const GraphElem* adj = node_adj(c, v, deg);  // any GPU of this node
    if (dcontains(adj, deg, w)) ++my;
  }
  my = cg::reduce(warp, my, cg::plus<VmmCount>());
  if (warp.thread_rank() == 0 && my) atomicAdd(count, my);
}

void vmm_count_recv(const NodeCSR& csr, const GraphElem* rbuf, GraphElem npairs,
                    VmmCount* d_count, cudaStream_t stream)
{
  if (npairs <= 0) return;
  k_count_recv<<<grid_for(npairs, VMM_BLOCK), VMM_BLOCK, 0, stream>>>(csr, rbuf, npairs, d_count);
  CudaCall(cudaGetLastError());
}
#endif
