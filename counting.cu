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
  //GraphElem nblocks = (end - start) > MAX_GRIDDIM ? MAX_GRIDDIM : (end - start); 
  //local_count_gpu_kernel<<<nblocks, GRIDDIM>>>(start, end, jstart, jend, d_edge_list, d_local_count);
  GraphElem range = std::ceil((end - start) / 2)*2;
  if (range < BLOCKDIM04)
    range = BLOCKDIM04;
  GraphElem nblocks = (range / BLOCKDIM04) > MAX_GRIDDIM ? MAX_GRIDDIM : (range / BLOCKDIM04);
    
  local_count_gpu_kernel<<<nblocks, BLOCKDIM04>>>(start, end, jstart, jend, d_edge_list, d_local_count);
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
  __shared__ GraphElem count;

  if (threadIdx.x == 0)
    count = 0;
  __syncthreads();

  for (GraphElem i = start + blockDim.x * blockIdx.x + threadIdx.x; i < end; i +=  blockDim.x*gridDim.x)
  {
    for(GraphElem k = jstart; k < jend; k++)
    {
      if (d_edge_list[k] == d_edge_list[i])
        count++;
    }
  }

  if (threadIdx.x == 0)
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


