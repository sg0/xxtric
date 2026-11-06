#pragma once
#ifndef COUNTING_CUH
#define COUNTING_CUH
#include <cuda.h>
#include <cuda_runtime.h>

#include "deftypes.hpp"

#ifdef DEBUG_CUDA
#define CudaCall(ans) { CudaAssert((ans), __FILE__, __LINE__); }
inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
   {
       std::cerr << "GPUassert: " << cudaGetErrorString(code)
                 << " " << file << " " << line << std::endl;
       if (abort)
           exit(code);
    }
 }
#else
#define CudaCall(ans) {ans;}
#endif

#ifdef DEBUG_CUDA
#define CudaLaunch(kernel) \
{ \
     kernel; \
     CudaCall(cudaPeekAtLastError()); \
     CudaCall(cudaDeviceSynchronize()); \
}
#else
#define CudaLaunch(kernel) \
{ \
     kernel; \
}
#endif


#define CudaMalloc(ptr,size) \
{\
   CudaCall(cudaMalloc((void**)&ptr,size));\
}
#define CudaMallocHost(ptr, size)\
{\
    CudaCall(cudaMallocHost((void**)&ptr, size)); \
}
#define CudaMallocManaged(ptr, size)\
{\
    CudaCall(cudaMallocManaged((void**)&ptr, size)); \
}

#define CudaStreamCreate(stream) \
{\
   CudaCall(cudaStreamCreate((cudaStream_t*)&stream));\
}

#define CudaStreamCreateNB(stream) \
{\
   CudaCall(cudaStreamCreateWithFlags((cudaStream_t*)&stream, cudaStreamNonBlocking));\
}

#define CudaStreamDestroy(stream) \
{\
   CudaCall(cudaStreamDestroy(stream));\
}

#define CudaMemcpyUVA(dest, src, size)\
CudaCall(cudaMemcpy(dest, src, size, cudaMemcpyDefault))

#define CudaMemcpyAsyncUVA(dest, src, size, stream)\
CudaCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDefault, stream))

#define CudaMemcpyHtoD(dest, src, size)\
CudaCall(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice))

#define CudaMemcpyDtoH(dest, src, size)\
CudaCall(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost))

#define CudaMemcpyAsyncHtoD(dest, src, size, stream) \
CudaCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, stream))

#define CudaMemcpyAsyncDtoH(dest, src, size, stream) \
CudaCall(cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost, stream))

#define CudaFree(ptr) \
{\
    if(ptr != nullptr) \
        CudaCall(cudaFree(ptr));\
    ptr = nullptr; \
}

#define CudaFreeHost(ptr) \
{\
   if(ptr != nullptr) \
       CudaCall(cudaFreeHost(ptr)); \
   ptr = nullptr; \
}

#define CudaStreamDestroy(stream) \
{\
   CudaCall(cudaStreamDestroy(stream));\
}

#define CudaStreamSynchronize(stream) \
{\
   CudaCall(cudaStreamSynchronize(stream));\
}

#define CudaDeviceSynchronize()\
CudaCall(cudaDeviceSynchronize())


#define CudaMemset(src, val, size) \
{\
   CudaCall(cudaMemset(src, val, size));\
}

#define CudaHostRegister(ptr, size, flag) \
{\
    CudaCall(cudaHostRegister(ptr, size, flag));\
}

#define CudaHostUnregister(ptr) \
{\
    CudaCall(cudaHostUnregister(ptr));\
}

#define CudaSetDevice(id) \
{\
   CudaCall(cudaSetDevice(id));\
}

#define CudaIPCOpenMemHandle(ptr, handle) \
{\
   CudaCall(cudaIpcOpenMemHandle(((void**)&ptr), handle, cudaIpcMemLazyEnablePeerAccess));\
}

#define CudaIPCCloseMemHandle(ptr) \
{\
   CudaCall(cudaIpcCloseMemHandle(((void**)&ptr)));\
}

#define CudaIPCGetMemHandle(ptr, handle) \
{\
   CudaCall(cudaIpcGetMemHandle(((cudaIpcMemHandle_t*)&handle), ptr));\
}

//define some cuda kernel configuration
#define TILESIZE00      (64)
#define TILESIZE01      (32)
#define TILESIZE02      (16)
#define TILESIZE03      (8)
#define TILESIZE04      (4)
#define TILESIZE05      (2)
#define BLOCKDIM00      (32)
#define BLOCKDIM01      (64)
#define BLOCKDIM02      (128)
#define BLOCKDIM03      (256)
#define BLOCKDIM04      (512)

#define GRIDDIM         (64)
#define MAX_BLOCKDIM    (1024)
#define MAX_GRIDDIM     (65535)

#ifndef MAX_CUDA_STREAMS
#define MAX_CUDA_STREAMS (8)
#endif

#define FULL_MASK      (0xFFFFFFFF) 

#ifdef USE_CG_CLUSTER_GROUP
#ifndef TB_PER_CLUSTER
#define TB_PER_CLUSTER (8)
#endif
#endif

__device__ 
GraphElem lower_bound
(
 const GraphElem size, 
 const GraphElem value, 
 const GraphElem* __restrict__ d_parts
);

__device__ bool edge_exists
(
 GraphElem e0, 
 GraphElem e1, 
 GraphElem se, 
 const GraphElem* __restrict__ d_edge_list
 );

__device__ void count_if_edge_exists
(
 GraphElem e0, 
 GraphElem e1, 
 GraphElem se, 
 const GraphElem* __restrict__ d_edge_list,
 GraphElem& t_count
 );

__device__ void count_if_edge_exists_smem
(
 GraphElem s1,
 GraphElem se, 
 volatile GraphElem* s_edge_list,
 GraphElem& t_count
 );

__device__ GraphElem next_power_of_2
(
 GraphElem x
);

__device__ 
void unroll_last_warp
(
 volatile GraphElem *sdata, 
 GraphElem tid
);

__device__ GraphElem midpoint
(
 GraphElem const& a, 
 GraphElem const& b
);

 void single_gpu_count_masked
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
);

 void single_gpu_count_masked
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
);

template <int tilesize, int blocksize>
__global__  
 void single_gpu_count_masked_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
);

void single_gpu_count
(
 const GraphElem nv,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
);

template <int tilesize, int blocksize>
__global__  
void single_gpu_count_kernel
(
 const GraphElem nv,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
);
 
void local_count_gpu_masked
(
 const GraphElem nedges,
 const GraphElem base,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
);

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
);
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
);
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
 );

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
 );

/// CUDA IPC variants

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
 );

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
 );

void local_count_gpu_masked_ext
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices_other,
 const GraphElem* __restrict__ d_edge_list_other,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
);

template <int tilesize, int blocksize>
__global__ void local_count_gpu_masked_ext_kernel
(
 const GraphElem nedges,
 const GraphElem* __restrict__ d_edge_mask,
 const GraphElem* __restrict__ d_indices_other,
 const GraphElem* __restrict__ d_edge_list_other,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count
);

/// Recv-side kernels
 void local_count_gpu_recv
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ d_rbuf,
 const GraphElem* __restrict__ d_indices,
 const GraphElem* __restrict__ d_edge_list,
 GraphElem* __restrict__ d_local_count,
 cudaStream_t stream
);

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
 );

/// CPU kernels
void local_count_cpu_inner
(
 GraphElem start,
 GraphElem end,
 GraphElem jstart,
 GraphElem jend,
 GraphElem* __restrict__ edge_list,
 GraphElem* d_local_count 
 );

void local_count_cpu_recv
(
 const GraphElem count,
 const GraphElem base,
 const GraphElem* __restrict__ rbuf,
 const GraphElem* __restrict__ indices,
 const GraphElem* __restrict__ edge_list,
 GraphElem* __restrict__ local_count
 );

#if defined(USE_CUDA_VMM_MPI)
// Kernels + host launchers for the VMM / GPU-buffered distributed triangle
// counting variant (xxtric-gpu-vmm-mpi.hpp).
struct NodeCSR
{
  int ngpus = 0;
  const GraphElem* vsplit = nullptr;       // [ngpus+1]
  const GraphElem* const* ind = nullptr;   // [ngpus]
  const GraphElem* const* lst = nullptr;   // [ngpus]
};

/// Remote-work plan of one GPU, built entirely on that GPU.
///
/// A "remote item" is an edge (u,v) owned by this GPU with u < v and v on a
/// different node; it produces cnt = |{n : lst[n] in [nlo, nlo+cnt)}| wedge
/// pairs (v, w) that the owner of v must check (is w a neighbour of v?).
/// Items are sorted by destination node, and an exclusive scan over the
/// counts (posx) gives every pair a unique position. Destination s owns the
/// contiguous position range [doff[s], doff[s] + dtot[s]) and the items
/// [dbeg[s], dbeg[s+1]). So any GPU thread can materialise pair #p of any
/// destination in O(log) without counters, atomics or host involvement.
struct VmmRemotePlan
{
  GraphElem  nrem  = 0;        // #remote items
  GraphElem  npairs = 0;       // #pairs over all destinations
  GraphElem* items = nullptr;  // [nrem]   edge index (local to this GPU), sorted by dest
  GraphElem* nlo   = nullptr;  // [nrem]   first n of the item's pair run
  GraphElem* posx  = nullptr;  // [nrem+1] exclusive scan of pair counts
  GraphElem* dbeg  = nullptr;  // [nparts+1] first item per destination
  GraphElem* doff  = nullptr;  // [nparts] first pair position per destination
  GraphElem* dtot  = nullptr;  // [nparts] #pairs per destination
};

/// One outgoing stream (this GPU -> destination node), as seen by the
/// pair-generation kernel. The stream is cut into rounds of `cap` pairs;
/// round k writes into set (k % nsets) of the slot's buffer region.
struct VmmSendSlot
{
  GraphElem dbeg, dend;  // item range
  GraphElem doff;        // first pair position
  GraphElem tot;         // #pairs in this stream
  GraphElem cap;         // pairs per round (<= tot)
  GraphElem bufoff;      // offset of set 0 in the GPU's comm chunk, in GraphElem units
  int nsets;             // 1 or 2 (double buffering when rounds > 1)
  int dest;              // destination node (MPI rank)
};

/// Triangle counters are always 64-bit unsigned (independent of GraphElem).
using VmmCount = unsigned long long;

// ---------------------------------------------------------------------------
// Host launchers (all asynchronous on `stream` unless noted). The caller must
// have made the GPU that owns `stream` current (cudaSetDevice).
// ---------------------------------------------------------------------------

/// erange[2u], erange[2u+1] = min/max neighbour of every vertex u owned by
/// GPU g (nv = #vertices of GPU g) with degree >= 2 (others are left untouched, i.e. 0/0 -> empty).
/// `erange` is a node-wide VMM pointer; stores may land on peer GPUs.
void vmm_build_erange(const NodeCSR& csr, int g, GraphElem nv, GraphElem* erange,
                      cudaStream_t stream);

/// Count every triangle u < v < w whose wedge (u,v,w) is fully node-local:
/// u owned by GPU g, v anywhere on this node (adjacency read via VMM).
void vmm_count_intranode(const NodeCSR& csr, int g, GraphElem ne,
                         VmmCount* d_count, cudaStream_t stream);

/// Build the remote plan for GPU g (classify -> compact -> sort -> scan).
/// Allocates the plan arrays on the current device. SYNCHRONOUS on `stream`
/// (it needs the number of remote items on the host to size allocations).
/// `erange` may be nullptr (no min/max pruning).
void vmm_plan_remote(const NodeCSR& csr, int g, GraphElem ne,
                     const GraphElem* d_parts, int nparts, int my_part,
                     const GraphElem* erange, VmmRemotePlan& plan,
                     cudaStream_t stream);

void vmm_plan_free(VmmRemotePlan& plan);

/// Write the pairs of round `round` of every slot into the comm buffer
/// `sbuf` (GPU g's chunk). `max_cap` = max over slots of cap.
void vmm_generate_pairs(const NodeCSR& csr, int g, const VmmRemotePlan& plan,
                        const VmmSendSlot* d_slots, int nslots, GraphElem max_cap,
                        int round, GraphElem* sbuf, cudaStream_t stream);

/// Check `npairs` received pairs (v, w), v owned by this node: count w in adj(v).
void vmm_count_recv(const NodeCSR& csr, const GraphElem* rbuf, GraphElem npairs,
                    VmmCount* d_count, cudaStream_t stream);
#endif
#endif
