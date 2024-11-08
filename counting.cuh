#include <cuda.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h> 

namespace cg = cooperative_groups;

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
 );

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
 );

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
 );

void local_count_cpu
(
 const GraphElem start,
 const GraphElem end,
 const GraphElem jstart,
 const GraphElem jend,
 GraphElem* __restrict__ edge_list,
 GraphElem* d_local_count 
 );
