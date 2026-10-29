#pragma once
#ifndef XXTRIC_CUDA_IPC_HPP
#define XXTRIC_CUDA_IPC_HPP

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string> 
#include <memory>

#include "counting.cuh"

class TriangleCounting
{
  public:

    TriangleCounting(Graph* g): 
      g_(g), ntriangles_(0), d_eind_(nullptr), d_elst_(nullptr)
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem lne = g_->get_lne();
    const GraphElem nv = g_->get_nv();

    std::vector<GraphElem> etails(lne);

    const GraphElem base = g_->get_base(rank_);

    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;

      for (GraphElem m = e0; m < e1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        etails[m] = edge_m.tail_;
      }
    }
    
    MPI_Barrier(comm_);

    /// assume all tasks/node can see all GPUs in a node
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank_%deviceCount);

    if (size_ > deviceCount)
    {
      if (rank_ == 0)
        std::cout << "*Error*: #Processes greater than available devices, IPC version cannot proceed!" << std::endl;          
      MPI_Abort(comm_, -99);
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (!prop.unifiedAddressing)
    {
      if (rank_ == 0)
        std::cout << "*Error*: GPU does not support unified addressing, IPC version cannot proceed!" << std::endl;          
      MPI_Abort(comm_, -99);
    }

    CudaMalloc(d_eind_, (lnv+1)*sizeof(GraphElem));
    CudaMalloc(d_elst_, lne*sizeof(GraphElem));
    CudaMalloc(d_ntriangles_, sizeof(GraphElem));

    CudaMemcpyHtoD(d_eind_, g_->edge_indices_.data(), (lnv+1)*sizeof(GraphElem));
    CudaMemcpyHtoD(d_elst_, etails.data(), lne*sizeof(GraphElem));
    CudaMemset(d_ntriangles_, 0, sizeof(GraphElem));

    /// Local counting in split mode
#if defined(USE_SEPARATE_KERNELS)
    std::vector<std::array<GraphElem,3>> edge_mask;
#endif
     
    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamCreate(streams[t]);
       
    cudaIpcMemHandle_t local_elst_handle, local_eind_handle;
    cudaIpcMemHandle_t *d_eind_handles = new cudaIpcMemHandle_t[size_];
    cudaIpcMemHandle_t *d_elst_handles = new cudaIpcMemHandle_t[size_];

    CudaIPCGetMemHandle(d_eind_, local_eind_handle);
    CudaIPCGetMemHandle(d_elst_, local_elst_handle);

    MPI_Allgather(&local_eind_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
        d_eind_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm_);
    MPI_Allgather(&local_elst_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
        d_elst_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm_);
   
    std::vector<GraphElem*> d_eind_nbr(size_, nullptr), d_elst_nbr(size_, nullptr);

#if !defined(USE_SEPARATE_KERNELS)
    GraphElem** d_edge_mask_nbr = new GraphElem*[size_];
    std::vector<std::vector<GraphElem>> v_edge_mask_nbr(size_);
#endif
    GraphElem* d_edge_mask = nullptr;
    std::vector<GraphElem> v_edge_mask;

    for (int t = 0; t < size_; t++)
    {    
      CudaIPCOpenMemHandle(d_eind_nbr[t], d_eind_handles[t]);
      CudaIPCOpenMemHandle(d_elst_nbr[t], d_elst_handles[t]);
    }

    MPI_Barrier(comm_);

    double t0 = MPI_Wtime();
    
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1, le0, le1;
      g_->edge_range(i, e0, e1);
      const GraphElem global_i = g_->local_to_global(i);

      if ((e0 + 1) == e1)
        continue;

      for (GraphElem m = e0; m < e1-1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        const int owner = g_->get_owner(edge_m.tail_);

        if (global_i >= edge_m.tail_)
          continue;

        if (owner == rank_)
        {
#if defined(USE_SEPARATE_KERNELS)
          g_->edge_range(g_->global_to_local(edge_m.tail_), le0, le1);
          edge_mask.push_back(std::array<GraphElem, 3>{e1, m, (le1-le0+1)});
#else
          v_edge_mask.push_back(e1);
          v_edge_mask.push_back(m);
#endif
        }
        else
        {
          if (owner > rank_)
          {
            const GraphElem base_m = edge_m.tail_-g_->get_base(owner);           
#if defined(USE_SEPARATE_KERNELS)
            local_count_gpu_ext(m+1, e1, base_m, (GraphElem*)d_eind_nbr[owner], d_elst_, (GraphElem*)d_elst_nbr[owner], d_ntriangles_, &streams[nst % MAX_CUDA_STREAMS]);
            nst++;
#else
            v_edge_mask_nbr[owner].push_back(e1);
            v_edge_mask_nbr[owner].push_back(m);
            v_edge_mask_nbr[owner].push_back(base_m);
#endif
          }
        }
      }
    }

#if defined(USE_SEPARATE_KERNELS)
    for (int t = 0; t < size_; t++)
    {    
      CudaIPCCloseMemHandle(d_eind_nbr[t]);
      CudaIPCCloseMemHandle(d_elst_nbr[t]);
    }
        
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamSynchronize(streams[t]);

    CudaDeviceSynchronize();

    d_eind_nbr.clear();
    d_elst_nbr.clear();
#else
    nst = 0;
#endif

#if defined(USE_SEPARATE_KERNELS)
    std::sort(edge_mask.begin(), edge_mask.end(), 
      [](auto a, auto b) { return a[2] > b[2]; });   
   
    // copy 50% of the smallest local edges in terms 
    // of neighborhood size 
    for (GraphElem i = 0; i < edge_mask.size()/2; i++)
    {
      v_edge_mask.push_back(edge_mask[i][0]);
      v_edge_mask.push_back(edge_mask[i][1]);
    }
    const GraphElem nedges = v_edge_mask.size()/2;
    CudaMalloc(d_edge_mask, nedges*2*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask, v_edge_mask.data(), nedges*2*sizeof(GraphElem));
    
    // bulk GPU kernel
    local_count_gpu_masked(nedges, base, d_edge_mask, d_eind_, d_elst_, d_ntriangles_, streams[0]);
    nst++;

    // remaining GPU kernels
    for (GraphElem i = edge_mask.size()/2; i < edge_mask.size(); i++)
    {
      GraphElem jstart, jend;
      Edge const& edge_m = g_->get_edge(edge_mask[i][1]);
      g_->edge_range(g_->global_to_local(edge_m.tail_), jstart, jend);

      local_count_gpu_inner(edge_mask[i][1]+1, edge_mask[i][0], jstart, jend, d_elst_, d_ntriangles_, &streams[nst % MAX_CUDA_STREAMS]);
      nst++;
    }
#else
    /// remote 
    for (int p = 0; p < size_; p++)
    {
      if (!v_edge_mask_nbr[p].empty())
      {
        const GraphElem nedges_rem = v_edge_mask_nbr[p].size()/3;
        CudaMalloc(d_edge_mask_nbr[p], nedges_rem*3*sizeof(GraphElem));
        CudaMemcpyHtoD(d_edge_mask_nbr[p], v_edge_mask_nbr[p].data(), nedges_rem*3*sizeof(GraphElem));

        local_count_gpu_masked_ext(nedges_rem, d_edge_mask_nbr[p], (GraphElem*)d_eind_nbr[p], (GraphElem*)d_elst_nbr[p], d_elst_, d_ntriangles_, streams[nst % MAX_CUDA_STREAMS]);
        nst++;
      }
    }

    /// local
    const GraphElem nedges = v_edge_mask.size()/2;
    CudaMalloc(d_edge_mask, nedges*2*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask, v_edge_mask.data(), nedges*2*sizeof(GraphElem));
   
    // grid-block over vertices
    local_count_gpu_masked(nedges, base, d_edge_mask, d_eind_, d_elst_, d_ntriangles_, streams[nst % MAX_CUDA_STREAMS]);
#endif

#if defined(USE_SEPARATE_KERNELS)
    edge_mask.clear();
#else    
    for (int t = 0; t < size_; t++)
    {    
      CudaIPCCloseMemHandle(d_eind_nbr[t]);
      CudaIPCCloseMemHandle(d_elst_nbr[t]);

      v_edge_mask_nbr[t].clear();
    }
        
    d_eind_nbr.clear();
    d_elst_nbr.clear();

    v_edge_mask_nbr.clear();
#endif
    
    v_edge_mask.clear();
    
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
    {
      CudaStreamSynchronize(streams[t]);
      CudaStreamDestroy(streams[t]);
    }

    CudaDeviceSynchronize();
    delete []streams;

        
    double t1 = MPI_Wtime(); 
    double p_tot = t1 - t0, t_tot = 0.0, t_max = 0.0, t_min = 0.0;

    MPI_Barrier(comm_);
    
    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
    MPI_Reduce(&p_tot, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
    MPI_Reduce(&p_tot, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm_);

    if (rank_ == 0) 
    {   
      std::cout << "Avg/Min/Max times for triangle counting including misc. operations (secs.): " 
        << ((double)(t_tot / (double)size_)) << ", " << t_min << ", " << t_max << std::endl;
    }
    
    delete []d_eind_handles;
    delete []d_elst_handles;

    CudaFree(d_edge_mask);

#if defined(USE_SEPARATE_KERNELS)
#else    
    for (int t = 0; t < size_; t++)
    {    
      if (d_edge_mask_nbr[t] != nullptr)
        CudaFree(d_edge_mask_nbr[t]);
    }
    delete []d_edge_mask_nbr;
#endif
  }

    ~TriangleCounting() {}

    void clear()
    {
      CudaFree(d_eind_);
      CudaFree(d_elst_);
      CudaFree(d_ntriangles_);
    }
    
    inline GraphElem count()
    {
      CudaDeviceSynchronize();
      CudaMemcpyDtoH(&ntriangles_, d_ntriangles_, sizeof(GraphElem));

      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
      
      return ttc;
    }

  private:
    Graph* g_;

    GraphElem ntriangles_;

    GraphElem *d_eind_, *d_elst_, *d_ntriangles_; // device

    int rank_, size_;
    MPI_Comm comm_;
};
#endif
