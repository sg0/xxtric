#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h> 

namespace cg = cooperative_groups;

#include "deftypes.hpp"

template<const int WarpSize, const int BlockSize>
__global__
void local_count_gpu
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 GraphElem* __restrict__ edge_list,
 GraphElem* d_local_count 
 )
{   
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);
    __shared__ GraphElem count;

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    if (threadIdx.x == 0) 
      count = 0;
    __syncthreads();

    GraphElem step = gridDim.x*BlockSize/WarpSize;
    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    
    for(GraphElem u = u0 + start; u < end; u += step)
    {   
        for(GraphElem e = jstart + warp_tid; e < jend; e += WarpSize)
        {
          if (edge_list[e] == edge_list[u])
            count++;
        }

        for(int i = warp.size()/2; i > 0; i/=2)
            count += warp.shfl_down(count, i);
 
        warp.sync();
    }

    if (block_tid == 0)
      *d_local_count += count; 
}

template<const int WarpSize, const int BlockSize>
__global__
void local_count_gpu1
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 GraphElem* __restrict__ edge_list,
 GraphElem* d_local_count 
 )
{   
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);

    const unsigned block_tid = block.thread_rank();
    const unsigned warp_tid = warp.thread_rank();

    GraphElem step = gridDim.x*BlockSize/WarpSize;
    GraphElem u0 = block_tid/WarpSize+BlockSize/WarpSize*blockIdx.x;
    
    for(GraphElem u = u0 + start; u < end; u += step)
    {   
        for(GraphElem e = jstart + warp_tid; e < jend; e += WarpSize)
        {
          if (edge_list[e] == edge_list[u])
            *d_local_count += 1;
        }

        for(int i = warp.size()/2; i > 0; i/=2)
            *d_local_count += warp.shfl_down(*d_local_count, i);
    }
}

template<const int blocksize>
__global__
  void local_count_gpu2
(
 GraphElem start,
 GraphElem end,
 GraphElem parent_edge,
 GraphElem jstart,
 GraphElem jend,
 GraphElem* __restrict__ edge_indices,
 GraphElem* __restrict__ edge_list,
 GraphElem* d_local_count 
 )
{
  volatile __shared__ short data[blocksize];
  __shared__ GraphElem count;
  GraphElem step = gridDim.x;
    
  if (threadIdx.x == 0) 
    count = 0;
  __syncthreads();

  for(GraphElem i = start + blockIdx.x; i < end; i += step)
  {
    if (parent_edge < edge_list[i])
    {
      for(GraphElem k = jstart + threadIdx.x; k < jend; k+=blocksize)
      {
        if (edge_list[k] == edge_list[i])
          count++;
      }

      data[threadIdx.x] = count;
      __syncthreads();

      for (unsigned int s=blocksize/2; s>=32; s>>=1)
      {
        if (threadIdx.x < s)
          data[threadIdx.x] = data[threadIdx.x+s];
        __syncthreads();
      }
      count = data[threadIdx.x%32];
      for (int offset = 16; offset > 0; offset /= 2)
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
  }

  if(threadIdx.x == 0)
    *d_local_count += count; 
  __syncthreads();
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

/*
  for (GraphElem k = 0; k < count;)
            {
              if (rbuf_[p*bufsize_+k] == -1)
              {
                k += 1;
                prev = k;
                continue;
              }

              tup[0] = rbuf_[p*bufsize_+k];
              GraphElem curr_count = 0;

              for (GraphElem m = k + 1; m < count; m++)
              {
                if (rbuf_[p*bufsize_+m] == -1)
                {
                  curr_count = m + 1;
                  break;
                }

                tup[1] = rbuf_[p*bufsize_+m];

#if defined(RECV_MAP)
                check_rmap(tup);
#else
#if defined(USE_OPENMP_NESTED)
                if (check_edgelist_omp(tup))
                  ntriangles_ += 1;
#else
#if defined(FAST_CHECK_EDGELIST)
                if (fast_check_edgelist(tup))
#else
                if (check_edgelist(tup))
#endif
                  ntriangles_ += 1;
#endif
#endif
                in_nghosts_ -= 1;
                recv_count_[source] -= 1;
              }

              k += (curr_count - prev);
              prev = k;
            }
*/


