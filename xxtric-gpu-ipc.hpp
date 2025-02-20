// ***********************************************************************
//
//                              TRIC
//
// ***********************************************************************
//
//       Copyright (2019) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 
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
      g_(g), erange_(nullptr), ntriangles_(0), d_eind_(nullptr), 
      d_elst_(nullptr), d_elst_handles_(nullptr), d_ntriangles_(nullptr) 
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem lne = g_->get_lne();
    const GraphElem nv = g_->get_nv();

    erange_ = new GraphElem[nv*2]();
    std::vector<GraphElem> etails(lne);

    double t0 = MPI_Wtime();

    // store edge ranges
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

      Edge const& edge_s = g_->get_edge(e0);
      Edge const& edge_t = g_->get_edge(e1-1);

      erange_[(i + base)*2] = edge_s.tail_;
      erange_[(i + base)*2+1] = edge_t.tail_;
    }
    
    MPI_Barrier(comm_);

    MPI_Allreduce(MPI_IN_PLACE, erange_, nv*2, MPI_GRAPH_TYPE, 
        MPI_SUM, comm_);
    
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
    std::vector<std::array<GraphElem,3>> edge_mask;
     
    /// manage streams 
    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamCreate(streams[t]);
        
    d_elst_handles_ = new cudaIpcMemHandle_t[size_];
    CudaIPCGetMemHandle(d_elst_, d_elst_handles_[rank_]);

    MPI_Allgather(&d_elst_handles_[rank_], sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
        d_elst_handles_, sizeof(cudaIpcMemHandle_t), MPI_BYTE, comm_);

    MPI_Barrier(comm_);

    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1, le0, le1, tup[2];
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
          g_->edge_range(g_->global_to_local(edge_m.tail_), le0, le1);
          edge_mask.push_back(std::array<GraphElem, 3>{e1, m, (le1-le0+1)});
        }
        else
        {
          if (owner > rank_)
          {
            GraphElem *owner_d_elst;
            GraphElem jstart, jend;

            CudaIPCOpenMemHandle(owner_d_elst, d_elst_handles_[owner]);
            Edge const& edge_m = g_->get_edge(m);
            g_->edge_range(g_->global_to_local(edge_m.tail_), jstart, jend);

            local_count_gpu_inner(m+1, e1, jstart, jend, owner_d_elst, d_ntriangles_, &streams[nst % MAX_CUDA_STREAMS]);
            nst++;

            CudaIPCCloseMemHandle(owner_d_elst);
          }
        }
      }
    }
        
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamSynchronize(streams[t]);

    CudaDeviceSynchronize();

    GraphElem* d_edge_mask = nullptr;
    std::vector<GraphElem> v_edge_mask;
    nst = 0;

#if defined(SORT_BY_NUM_EDGES)
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
    local_count_gpu_masked3(nedges, base, d_edge_mask, d_eind_, d_elst_, d_ntriangles_, streams[0]);
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
    for (GraphElem i = 0; i < edge_mask.size(); i++)
    {
      if ((edge_mask[i][0]-edge_mask[i][1]+1) > edge_mask[i][2]) // e1-m > #nbrs(m)
      {
        v_edge_mask.push_back(edge_mask[i][0]);
        v_edge_mask.push_back(edge_mask[i][1]);
      }
      else // grid-block over edge-list
      {
        GraphElem jstart, jend;
        Edge const& edge_m = g_->get_edge(edge_mask[i][1]);
        g_->edge_range(g_->global_to_local(edge_m.tail_), jstart, jend);

        local_count_gpu_inner(edge_mask[i][1]+1, edge_mask[i][0], jstart, jend, d_elst_, d_ntriangles_, &streams[nst % MAX_CUDA_STREAMS]);
        nst++;
      }
    }

    const GraphElem nedges = v_edge_mask.size()/2;
    CudaMalloc(d_edge_mask, nedges*2*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask, v_edge_mask.data(), nedges*2*sizeof(GraphElem));
    
    // grid-block over vertices
    local_count_gpu_masked3(nedges, base, d_edge_mask, d_eind_, d_elst_, d_ntriangles_, streams[nst % MAX_CUDA_STREAMS]);
#endif
    
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
    {
      CudaStreamSynchronize(streams[t]);
      CudaStreamDestroy(streams[t]);
    }

    CudaDeviceSynchronize();
    delete []streams;

    v_edge_mask.clear();
    edge_mask.clear();
        
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

    t0 = MPI_Wtime();

    
#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
    
    CudaFree(d_edge_mask);
  }

    ~TriangleCounting() {}

    void clear()
    {
      delete []erange_;
      delete []d_elst_handles_;
            
      CudaFree(d_eind_);
      CudaFree(d_elst_);
      CudaFree(d_ntriangles_);
    }
    
    inline bool edge_between_range(GraphElem const& x, GraphElem const& y) const
    {
      if ((y >= erange_[x*2]) && (y <= erange_[x*2+1]))
        return true;
      return false;
    }
         
    inline bool edge_above_min(GraphElem x, GraphElem y) const
    {
      if (y >= erange_[x*2])
        return true;
      return false;
    }

    inline bool edge_within_max(GraphElem x, GraphElem y) const
    {
      if (y <= erange_[x*2+1])
        return true;
      return false;
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
    GraphElem *erange_;

    GraphElem *d_eind_, *d_elst_, *d_ntriangles_; // device
    cudaIpcMemHandle_t *d_elst_handles_;

    int rank_, size_;
    MPI_Comm comm_;
};
#endif
