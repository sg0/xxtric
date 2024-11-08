
#pragma once
#ifndef COUNTING_CUH
#define COUNTING_CUH

#ifdef USE_CUDA
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

#define CudaDeviceSynchronize()\
CudaCall(cudaDeviceSynchronize())


#define CudaMemset(src, val, size) \
{\
   CudaCall(cudaMemset(src, val, size));\
}

#define CudaHostRegister(ptr, size, flag)\
{\
    CudaCall(cudaHostRegister(ptr, size, flag)); \
}

#define CudaHostUnregister(ptr)\
{\
    CudaCall(cudaHostUnregister(ptr));\
}

#define CudaSetDevice(id) \
{\
   CudaCall(cudaSetDevice(id));\
}

//define some cuda kernel configuration
#define TILESIZE01      (32)
#define TILESIZE02      (16)
#define TILESIZE03      (8)
#define TILESIZE04      (4)
#define TILESIZE05      (2)
#define WARPSIZE        (32)
#define BLOCKDIM01      (64)
#define BLOCKDIM02      (128)
#define BLOCKDIM03      (256)
#define BLOCKDIM04      (512)

#define GRIDDIM         (64)
#define MAX_BLOCKDIM    (1024)
#define MAX_GRIDDIM     (65535)
#endif // end of USE_CUDA



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

#endif
