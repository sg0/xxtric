#pragma once
#ifndef TENSOR_CUH
#define TENSOR_CUH

#define TENSORSHAPE00      (4096)
#define TENSORSHAPE01      (2048)
#define TENSORSHAPE02      (1024)
#define TENSORSHAPE03      (512)
#define TENSORSHAPE04      (256)
#define TENSORSHAPE05      (128)
#define TENSORSHAPE06      (64)
#define TENSORSHAPE07      (32)
#define TENSORSHAPE08      (16)
#define TENSORSHAPE09      (8)
#define TENSORSHAPE10      (4)

#include <cuda_runtime_api.h>
#include <mma.h>
using namespace nvcuda;

__device__ 
void count_if_edge_exists_wmma
(
 GraphElem s1,
 GraphElem se, 
 // wmma::fragment<wmma::accumulator, 16, 16, 16, int>& acc_frag,
 volatile int* t_frag,
 GraphElem& t_count
 );

 void single_gpu_count_wmma
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
 );

template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_wmma_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
);

#endif
