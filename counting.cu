#include "deftypes.hpp"
#include "counting.cuh"
#include <cassert>
#include <iostream>

#include <cooperative_groups.h> 
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
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
  const GraphElem nblocks = (nedges + blocksize - 1) / blocksize;

#if defined(USE_CG_CLUSTER_GROUP)
  int clustersize;
  for (int k = 1; k <= std::max(TB_PER_CLUSTER,1); k*=2)
  {
    if (nblocks % k == 0)
      clustersize = k;
  }
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
    config.dynamicSmemBytes = clustersize*sizeof(GraphElem);

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

  const GraphElem cluster_size = cluster.dim_blocks().x;
  const GraphElem cluster_tid = cluster.block_rank();

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
    
  count = cg::reduce(warp, count, cg::plus<GraphElem>());
  
  const GraphElem num_warps = BlockSize/WarpSize;
  const GraphElem warp_id = block_tid / WarpSize;
	__shared__ GraphElem block_smem[num_warps]; 
  extern __shared__ GraphElem cluster_smem[];
   
  if (warp_tid == 0)
    block_smem[warp_id] = count;
  
  block.sync();

  // block-level reduction
  GraphElem block_sum = 0;

  if (warp_id == 0)
  {
    GraphElem warp_result = (warp_tid < num_warps) ? block_smem[warp_tid] : 0;
    block_sum = cg::reduce(warp, warp_result, cg::plus<GraphElem>());
  }

  // cluster level reduction
  if (block_tid == 0)
    block_smem[0] = block_sum;

  block.sync();
	cluster.sync(); 

  if (block_tid == 0) 
  {
    GraphElem cluster_sum = 0;

#pragma unroll
    for (int b = 0; b < cluster_size; ++b) 
    {
      GraphElem* remote = cluster.map_shared_rank(block_smem, b);  
      cluster_sum += remote[0];
    }

    if (cluster_tid == 0)
      atomicAdd((unsigned long long int*)d_local_count, cluster_sum);
  }
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
