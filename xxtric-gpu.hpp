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
#ifndef XXTRIC_GPU_HPP
#define XXTRIC_GPU_HPP

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

#ifndef TAG_DATA
#define TAG_DATA  (100)
#endif

#include "counting.cuh"

#if defined(USE_SINGLE_DEVICE)
class TriangleCountingSingle
{
  public:

    TriangleCountingSingle(Graph* g): 
      g_(g), ntriangles_(0), d_eind_(nullptr), d_elst_(nullptr), d_ntriangles_(nullptr)
#if defined(SET_VERTEX_RANGE)
      ,mnv_(-1), d_ebeg_(nullptr)
#elif defined(SET_VERTEX_RANGE_SPLIT)
      ,edge_mask_(0), d_edge_mask_(nullptr), v_edge_mask_(0)
#endif
  {
    const GraphElem ne = g_->get_ne();
    const GraphElem nv = g_->get_nv();

    int deviceCount, device_id = 0;
    cudaDeviceProp prop;
    
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(device_id);

#if defined(CUDA_SET_L2_CACHE_LIMIT)
    GraphWeight c_factor = 0.75;
    cudaGetDeviceProperties(&prop, device_id);
    size_t l2_size = MIN(int(prop.l2CacheSize * c_factor), prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size);
#endif
#if defined(CUDA_SET_L2_CACHE_GRANULARITY)
    cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 128);
#endif

#if defined(SET_VERTEX_RANGE)
    std::vector<GraphElem> new_erange(nv*3, -1), upd_erange;
#endif

    std::vector<GraphElem> etails(ne);
#if defined(SHOW_GRAPH_STATS)
    std::vector<GraphElem> erange(nv);
#endif
    const GraphElem avgdeg = (ne / nv);

#if defined(USE_OPENMP)
#if defined(SET_VERTEX_RANGE_SPLIT)
#pragma omp declare reduction(merge : std::vector<GraphElem> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(merge : std::vector<std::array<GraphElem, 3>> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp parallel for firstprivate(avgdeg) reduction(merge: v_edge_mask_, edge_mask_) 
#else
#pragma omp parallel for  
#endif
#endif
    for (GraphElem i = 0; i < nv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;

#if defined(SHOW_GRAPH_STATS)
      erange[i] = e1 - e0;
#endif
      for (GraphElem m = e0; m < e1; m++)
      {
        Edge const& edge_m = g_->get_edge(m);
        etails[m] = edge_m.tail_;

        if (i >= edge_m.tail_)
          continue;

#if defined(SET_VERTEX_RANGE) 
        if (new_erange[i*3] == -1)
        {
          new_erange[i*3]   = i;
          new_erange[i*3+1] = m;
          new_erange[i*3+2] = m;
        }
        else
          new_erange[i*3+2] = m;
#elif defined(SET_VERTEX_RANGE_SPLIT)
        if (m != (e1-1))
        {
          GraphElem le0, le1, nbrs_m;
          g_->edge_range(edge_m.tail_, le0, le1);
          nbrs_m = (le1-le0+1);

#if defined(SORT_BY_NUM_EDGES)
          edge_mask_.push_back(std::array<GraphElem, 3>{e1, m, (e1-m+1)});
#else
          //if ((e1-m+1) < avgdeg) // avg. degree
          //if ((e1-m+1) > nbrs_m) // e1-m > #nbrs(m)
          /*
          if ((e1-m+1) < nbrs_m) // e1-m < #nbrs(m)
          {
            v_edge_mask_.push_back(e1);
            v_edge_mask_.push_back(m);
          }
          else
            edge_mask_.push_back(std::array<GraphElem, 3>{e1, m, (e1-m+1)});
          */  
          edge_mask_.push_back(std::array<GraphElem, 3>{e1, m, -1});
          //edge_mask_.push_back(std::array<GraphElem, 3>{e1, m, (e1-m+1)});
#endif
        }
#else
#endif
      }
    }

#if defined(SHOW_GRAPH_STATS)
    print_graph_stats(erange, nv, ne);
#endif

    CudaMalloc(d_eind_, (nv+1)*sizeof(GraphElem));
    CudaMalloc(d_elst_, ne*sizeof(GraphElem));
    CudaMalloc(d_ntriangles_, sizeof(GraphElem));

    CudaMemcpyHtoD(d_eind_, g_->edge_indices_.data(), (nv+1)*sizeof(GraphElem));
    CudaMemcpyHtoD(d_elst_, etails.data(), ne*sizeof(GraphElem));

#if defined(SET_VERTEX_RANGE_SPLIT)
#if defined(SORT_BY_NUM_EDGES)
    std::sort(edge_mask_.begin(), edge_mask_.end(), 
      [](auto a, auto b) { return a[2] > b[2]; });  

    const GraphWeight factor = 0.999;
    assert(factor <= 1.0);
    const GraphElem cutoff_n = (GraphElem)((GraphWeight)edge_mask_.size()*factor);

    for (GraphElem i = 0; i < cutoff_n; i++)
    {
      v_edge_mask_.push_back(edge_mask_[i][0]);
      v_edge_mask_.push_back(edge_mask_[i][1]);
    }

    // copy x% of the smallest local edges in terms 
    // of neighborhood size 
    const GraphElem nedges = v_edge_mask_.size()/2;
    CudaMalloc(d_edge_mask_, nedges*2*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask_, v_edge_mask_.data(), nedges*2*sizeof(GraphElem));
#else
    for (GraphElem i = 0; i < edge_mask_.size(); i++)
    {
      v_edge_mask_.push_back(edge_mask_[i][0]);
      v_edge_mask_.push_back(edge_mask_[i][1]);
    }
    const GraphElem nedges = v_edge_mask_.size()/2;
    CudaMalloc(d_edge_mask_, nedges*2*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask_, v_edge_mask_.data(), nedges*2*sizeof(GraphElem));
#endif
#endif

#if defined(SET_VERTEX_RANGE)
    upd_erange.reserve(nv*3);

    // erase entries with -1
    for (GraphElem n = 0; n < 3*nv; n+=3)
    {
      if (new_erange[n] == -1)
        continue;
      else
      {
        upd_erange.push_back(new_erange[n]);
        upd_erange.push_back(new_erange[n+1]);
        upd_erange.push_back(new_erange[n+2]);
      }
    }
    
    mnv_ = upd_erange.size()/3;
    
    CudaMalloc(d_ebeg_, 3*mnv_*sizeof(GraphElem));
    CudaMemcpyHtoD(d_ebeg_, upd_erange.data(), 3*mnv_*sizeof(GraphElem));
#endif

    CudaMemset(d_ntriangles_, 0, sizeof(GraphElem));

    etails.clear();
#if defined(SHOW_GRAPH_STATS)
    erange.clear();
#endif
#if defined(SET_VERTEX_RANGE)
    new_erange.clear();
    upd_erange.clear();
#endif
  }

#if defined(SHOW_GRAPH_STATS)
    void print_graph_stats(std::vector<GraphElem> const& vdeg, const GraphElem nv, const GraphElem ne)
    {
      GraphElem sumdeg = 0, maxdeg = 0, sdevs = 0;
      sumdeg = std::reduce(vdeg.begin(), vdeg.end());
      auto max_deg = std::max_element(vdeg.begin(), vdeg.end());
      maxdeg = *max_deg;

      GraphWeight average  = (GraphWeight) sumdeg / nv;
       
      for (auto d : vdeg)
      {
        GraphElem d_sq = d - average;
        sdevs += d_sq*d_sq;
      }

      GraphWeight var = (GraphWeight) sdevs / nv; 
      GraphWeight stddev   = sqrt(var);
      
      std::cout << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
      std::cout << "Graph edge distribution characteristics" << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
      std::cout << "Number of vertices: " << nv << std::endl;
      std::cout << "Number of edges: " << ne << std::endl;
      std::cout << "Maximum number of edges: " << maxdeg << std::endl;
      std::cout << "Average number of edges: " << average << std::endl;
      std::cout << "Variance: " << var << std::endl;
      std::cout << "Standard deviation: " << stddev << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
    }
#endif

    ~TriangleCountingSingle() {}

    GraphElem count()
    {
      const GraphElem nv = g_->get_nv();
#if defined(USE_SPLIT_KERNELS)
    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamCreateNB(streams[t]);

#if defined(USE_OPENMP)
#pragma omp parallel for
#endif
    for (GraphElem i = 0; i < nv; i++)
    {
      GraphElem e0, e1;
      GraphElem jstart, jend, id;

      g_->edge_range(i, e0, e1);

      if ((e0 + 1) != e1)
      {
        for (GraphElem m = e0; m < e1-1; m++)
        {
          Edge const& edge_m = g_->get_edge(m);

          if (i >= edge_m.tail_)
            continue;

          g_->edge_range(edge_m.tail_, jstart, jend);

#pragma omp atomic capture
          id = nst++;
          local_count_gpu_inner(m+1, e1, jstart, jend, d_elst_, d_ntriangles_, &streams[id % MAX_CUDA_STREAMS]);
        }
      }
    }

    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamSynchronize(streams[t]);
    delete []streams;
#elif defined(SET_VERTEX_RANGE)
    single_gpu_count(mnv_, d_eind_, d_ebeg_, d_elst_, d_ntriangles_);
#elif defined(SET_VERTEX_RANGE_SPLIT)
#if defined(SORT_BY_NUM_EDGES)
    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamCreateNB(streams[t]);

    // bulk GPU kernel
    single_gpu_count_masked(v_edge_mask_.size()/2, d_edge_mask_, d_eind_, d_elst_, d_ntriangles_, streams[0]);

#if defined(USE_OPENMP)
#pragma omp parallel for 
#endif
    for (GraphElem i = v_edge_mask_.size()/2; i < edge_mask_.size(); i++)
    {
      GraphElem jstart, jend, id;
      Edge const& edge_m = g_->get_edge(edge_mask_[i][1]);
      g_->edge_range(g_->global_to_local(edge_m.tail_), jstart, jend);

#pragma omp atomic capture
      id = nst++;
      local_count_gpu_inner(edge_mask_[i][1]+1, edge_mask_[i][0], jstart, jend, d_elst_, d_ntriangles_, &streams[id % MAX_CUDA_STREAMS]);
    }
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamSynchronize(streams[t]);
    delete []streams;
#else
    // grid-block over vertices
    ///single_gpu_count_masked(v_edge_mask_.size()/2, d_edge_mask_, d_eind_, d_elst_, d_ntriangles_, streams[0]);
    single_gpu_count_masked(v_edge_mask_.size()/2, d_edge_mask_, d_eind_, d_elst_, d_ntriangles_);
    /*
    // grid-block over edge-list
    GraphElem jstart, jend, id;
#if defined(USE_OPENMP)
#pragma omp parallel for private(jstart, jend, id) default(shared)
#endif
    for (GraphElem i = 0; i < edge_mask_.size(); i++)
    {
      Edge const& edge_m = g_->get_edge(edge_mask_[i][1]);
      g_->edge_range(edge_m.tail_, jstart, jend);
#pragma omp atomic capture
      id = nst++;
      local_count_gpu_inner(edge_mask_[i][1]+1, edge_mask_[i][0], jstart, jend, d_elst_, d_ntriangles_, &streams[id % MAX_CUDA_STREAMS]);
    }
    */
#endif

#else
    single_gpu_count(nv, d_eind_, d_elst_, d_ntriangles_);
#endif

    CudaDeviceSynchronize();  
    CudaMemcpyDtoH(&ntriangles_, d_ntriangles_, sizeof(GraphElem));

    return ntriangles_;
    }

    void clear()
    {
      CudaFree(d_eind_);
      CudaFree(d_elst_);
      CudaFree(d_ntriangles_);

#if defined(SET_VERTEX_RANGE)
      CudaFree(d_ebeg_);
#elif defined(SET_VERTEX_RANGE_SPLIT)
      edge_mask_.clear();
      CudaFree(d_edge_mask_);
      v_edge_mask_.clear();
#else
#endif
    }

private:
    Graph* g_;
    GraphElem ntriangles_;
    GraphElem *d_eind_, *d_elst_, *d_ntriangles_; // device
#if defined(SET_VERTEX_RANGE)
    GraphElem mnv_;
    GraphElem *d_ebeg_;
#elif defined(SET_VERTEX_RANGE_SPLIT)
    std::vector<std::array<GraphElem,3>> edge_mask_;
    GraphElem *d_edge_mask_;
    std::vector<GraphElem> v_edge_mask_;
#else
#endif
};
#else // MPI-based distributed-memory counting
class TriangleCounting
{
  public:

    TriangleCounting(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), pdegree_(0), 
      sreq_(nullptr), erange_(nullptr), vcount_(nullptr), ntriangles_(0), 
      nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), 
      prev_k_(nullptr), stat_(nullptr), targets_(0), sources_(0), bufsize_(0), rreq_(nullptr),
      recv_count_(nullptr), rindex_(0), rdegree_(0), d_eind_(nullptr), d_elst_(nullptr), 
      d_rbuf_(nullptr), d_ntriangles_(nullptr)
#if defined(USE_CUDA_AWARE_MPI)
      , d_sbuf_(nullptr)
#else
      , rbuf_(nullptr)
#endif
#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
      , e_rstreams_(nullptr), count_rstreams_(0)
#endif
  {
    comm_ = g_->get_comm();
    MPI_Comm_size(comm_, &size_);
    MPI_Comm_rank(comm_, &rank_);

    const GraphElem lnv = g_->get_lnv();
    const GraphElem lne = g_->get_lne();
    const GraphElem nv = g_->get_nv();

    vcount_ = new GraphElem[lnv]();
    erange_ = new GraphElem[nv*2]();
    GraphElem *send_count  = new GraphElem[size_]();
    recv_count_  = new GraphElem[size_]();
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

#if defined(LESS_GLOBALMEM_READS) && !defined(COMBINED_LOCAL_COUNTING_KERNEL)
#else
    GraphElem* d_range = nullptr;
    CudaMalloc(d_range, 2*lnv*sizeof(GraphElem));
#endif
    CudaMalloc(d_eind_, (lnv+1)*sizeof(GraphElem));
    CudaMalloc(d_elst_, lne*sizeof(GraphElem));
    CudaMalloc(d_ntriangles_, sizeof(GraphElem));

    CudaMemcpyHtoD(d_eind_, g_->edge_indices_.data(), (lnv+1)*sizeof(GraphElem));
    CudaMemcpyHtoD(d_elst_, etails.data(), lne*sizeof(GraphElem));
#if defined(LESS_GLOBALMEM_READS) && !defined(COMBINED_LOCAL_COUNTING_KERNEL)
#else
    CudaMemcpyHtoD(d_range, &erange_[2*base], 2*lnv*sizeof(GraphElem));
#endif
    
    CudaMemset(d_ntriangles_, 0, sizeof(GraphElem));

    /// Local counting

#if defined(COMBINED_LOCAL_COUNTING_KERNEL)
    #pragma message "COMBINED_LOCAL_COUNTING_KERNEL has bugs; do not use!"
    GraphElem* d_vcount = nullptr;
    GraphElem* d_send_count = nullptr;
    GraphElem* d_parts = nullptr;

    CudaMalloc(d_vcount, lnv*sizeof(GraphElem));
    CudaMalloc(d_send_count, size_*sizeof(GraphElem));
    CudaMalloc(d_parts, (size_+1)*sizeof(GraphElem));
    
    CudaMemcpyHtoD(d_parts, g_->get_parts(), (size_+1)*sizeof(GraphElem));
    
    CudaMemset(d_vcount, 0, lnv*sizeof(GraphElem));
    CudaMemset(d_send_count, 0, size_*sizeof(GraphElem));

    combined_local_count_gpu(rank_, size_, lnv, d_parts, d_range, d_eind_, d_elst_, d_vcount, d_send_count, d_ntriangles_);
#elif defined(SEPARATE_LOCAL_COUNTING_KERNEL)    
#if defined(LESS_GLOBALMEM_READS)
    local_count_gpu(lnv, base, g_->get_bound(rank_), d_eind_, d_elst_, d_ntriangles_);
#else
    local_count_gpu(lnv, base, g_->get_bound(rank_), d_range, d_eind_, d_elst_, d_ntriangles_);
#endif
    /// Data requirements for remote counting
#if defined(USE_OPENMP)
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;

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

        if (owner > rank_)
        {
          for (GraphElem n = m + 1; n < e1; n++)
          {
            Edge const& edge_n = g_->get_edge(n);

            if (!edge_within_max(edge_m.tail_, edge_n.tail_))
              break;

            if (!edge_above_min(edge_m.tail_, edge_n.tail_))
              continue;

#if defined(USE_OPENMP)
#pragma omp atomic update
#endif
            send_count[owner] += 2;
            vcount_[i] += 1;
          }
        }
      }
    }
#elif defined(MASKED_LOCAL_COUNTING_KERNEL)
    std::vector<GraphElem> local_edge_range(lnv*3,-1);
#if defined(USE_OPENMP)
#pragma omp parallel for schedule(dynamic) default(shared)
#endif
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1, tup[2];
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
          if (local_edge_range[i*3] == -1)
          {
            local_edge_range[i*3]   = i;
            local_edge_range[i*3+1] = m;
            local_edge_range[i*3+2] = m;
          }
          else
            local_edge_range[i*3+2] = m;
        }
        else
        {
          if (owner > rank_)
          {
            for (GraphElem n = m + 1; n < e1; n++)
            {
              Edge const& edge_n = g_->get_edge(n);

              if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                break;
              if (!edge_above_min(edge_m.tail_, edge_n.tail_))
                continue;
#if defined(USE_OPENMP)
#pragma omp atomic update
#endif
              send_count[owner] += 2;
              vcount_[i] += 1;
            }
          }
        }
      }
    }

    GraphElem* d_edge_mask = nullptr;
    std::vector<GraphElem> edge_range;
    edge_range.reserve(lnv*3);

    // erase entries with -1
    // TODO FIXME reduce redundancies
    for (GraphElem n = 0; n < 3*lnv; n+=3)
    {
      if (local_edge_range[n] == -1)
        continue;
      else
      {
        edge_range.push_back(local_edge_range[n]);
        edge_range.push_back(local_edge_range[n+1]);
        edge_range.push_back(local_edge_range[n+2]);
      }
    }
    
    const GraphElem llnv = edge_range.size()/3;
    CudaMalloc(d_edge_mask, 3*llnv*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask, edge_range.data(), 3*llnv*sizeof(GraphElem));

#if defined(LESS_GLOBALMEM_READS)
    local_count_gpu_masked2(llnv, base, d_edge_mask, d_eind_, d_elst_, d_ntriangles_);
#else
    local_count_gpu_masked(llnv, base, d_range, d_edge_mask, d_eind_, d_elst_, d_ntriangles_);
#endif

    local_edge_range.clear();
    edge_range.clear();
#elif defined(SPLIT_LOCAL_COUNTING_KERNEL)
    std::vector<std::array<GraphElem,3>> edge_mask;
    
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
            for (GraphElem n = m + 1; n < e1; n++)
            {
              Edge const& edge_n = g_->get_edge(n);

              if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                break;
              if (!edge_above_min(edge_m.tail_, edge_n.tail_))
                continue;

              send_count[owner] += 2;
              vcount_[i] += 1;
            }
          }
        }
      }
    }
    
    /// manage streams
    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      cudaStreamCreate(&streams[t]);
    
    GraphElem* d_edge_mask = nullptr;
    std::vector<GraphElem> v_edge_mask;

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
      cudaStreamDestroy(streams[t]);
    delete []streams;

    v_edge_mask.clear();
    edge_mask.clear();
#else
    /// manage streams
    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      cudaStreamCreate(&streams[t]);

    for (GraphElem i = 0; i < nv; i++)
    {
      GraphElem e0, e1v;
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
          GraphElem jstart, jend;
          g_->edge_range(g_->global_to_local(edge_m.tail_), jstart, jend);
          local_count_gpu_inner(m+1, e1, jstart, jend, d_elst_, d_ntriangles_, &streams[nst % MAX_CUDA_STREAMS]);
          nst++;
        }
        else
        {
          if (owner > rank_)
          {
            for (GraphElem n = m + 1; n < e1; n++)
            {
              Edge const& edge_n = g_->get_edge(n);

              if (!edge_within_max(edge_m.tail_, edge_n.tail_))
                break;
              if (!edge_above_min(edge_m.tail_, edge_n.tail_))
                continue;

              send_count[owner] += 2;
              vcount_[i] += 1;
            }
          }
        }
      }
    }
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      cudaStreamDestroy(streams[t]);
    delete []streams;
#endif
        
    CudaDeviceSynchronize();
#if defined(COMBINED_LOCAL_COUNTING_KERNEL)
    CudaMemcpyDtoH(vcount_, d_vcount, lnv*sizeof(GraphElem));
    CudaMemcpyDtoH(send_count, d_send_count, size_*sizeof(GraphElem));

    CudaFree(d_vcount);
    CudaFree(d_send_count);
#endif

    double t1 = MPI_Wtime(); 
    double p_tot = t1 - t0, t_tot = 0.0, t_max = 0.0, t_min = 0.0;

    MPI_Barrier(comm_);
    
    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
    MPI_Reduce(&p_tot, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
    MPI_Reduce(&p_tot, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm_);

    if (rank_ == 0) 
    {   
      std::cout << "Avg/Min/Max times for local counting during instantiation (secs.): " 
        << ((double)(t_tot / (double)size_)) << ", " << t_min << ", " << t_max << std::endl;
    }

    t0 = MPI_Wtime();

    // outgoing/incoming data and buffer size
    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count_, 1, MPI_GRAPH_TYPE, comm_);
    
    for (GraphElem p = 0; p < size_; p++)
    {
      out_nghosts_ += send_count[p];
      in_nghosts_ += recv_count_[p];

      if (send_count[p] > 0)
        targets_.push_back(p);

      if (recv_count_[p] > 0)
        sources_.push_back(p);
    }
      
    pdegree_ = targets_.size();
    rdegree_ = sources_.size();

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], i});

    for (int i = 0; i < rdegree_; i++)
      rindex_.insert({sources_[i], i});

    nghosts_ = out_nghosts_ + in_nghosts_;

    bufsize_ = (nghosts_ < bufsize) ? nghosts_ : bufsize;

    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    if (rank_ == 0)
      std::cout << "Adjusted Per-PE buffer count: " << bufsize_ << std::endl;

    sbuf_ctr_ = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    sreq_     = new MPI_Request[pdegree_];
    stat_     = new char[pdegree_];
    rreq_     = new MPI_Request[rdegree_];

    std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
    std::fill(prev_k_, prev_k_ + pdegree_, -1);
    std::fill(prev_m_, prev_m_ + pdegree_, -1);
    std::fill(stat_, stat_ + pdegree_, '0');
    std::fill(rreq_, rreq_ + rdegree_, MPI_REQUEST_NULL);

#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
    sbuf_     = new GraphElem[pdegree_*bufsize_];
    rbuf_     = new GraphElem[rdegree_*bufsize_];
    count_rstreams_ = MIN(MAX_CUDA_STREAMS, rdegree_);
    e_rstreams_ = new cudaStream_t[count_rstreams_];
    for (int t = 0; t < count_rstreams_; t++)
      cudaStreamCreate(&e_rstreams_[t]);
    CudaMalloc(d_rbuf_, count_rstreams_*bufsize_*sizeof(GraphElem));
#elif defined(USE_CUDA_AWARE_MPI)
    CudaMallocHost(sbuf_, pdegree_*bufsize_*sizeof(GraphElem));
    CudaMalloc(d_sbuf_, pdegree_*bufsize_*sizeof(GraphElem));
    CudaMalloc(d_rbuf_, rdegree_*bufsize_*sizeof(GraphElem));
#else
    sbuf_     = new GraphElem[pdegree_*bufsize_];
    rbuf_     = new GraphElem[rdegree_*bufsize_];
    CudaMalloc(d_rbuf_, bufsize_*sizeof(GraphElem));
#endif

    MPI_Barrier(comm_);

#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
    
    delete []send_count;
#if defined(LESS_GLOBALMEM_READS) && !defined(COMBINED_LOCAL_COUNTING_KERNEL)
#else
    CudaFree(d_range);
#endif
#if defined(MASKED_LOCAL_COUNTING_KERNEL) || defined(SPLIT_LOCAL_COUNTING_KERNEL)
    CudaFree(d_edge_mask);
#endif
  }

    ~TriangleCounting() {}

    void clear()
    {
#if defined(USE_CUDA_AWARE_MPI)
      CudaFree(sbuf_);
      CudaFree(d_sbuf_);
#else
      delete []sbuf_;
      delete []rbuf_;
#endif
      delete []recv_count_;
      delete []sbuf_ctr_;
      delete []sreq_;
      delete []rreq_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;

      CudaFree(d_eind_);
      CudaFree(d_elst_);
      CudaFree(d_rbuf_);
      CudaFree(d_ntriangles_);

#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
      delete []e_rstreams_;
#endif
      pindex_.clear();
      targets_.clear();
      rindex_.clear();
      sources_.clear();
    }

    inline void nbsend(int owner)
    {
      if (stat_[pindex_[owner]] == '0' && sbuf_ctr_[pindex_[owner]] > 0)
      {
#if defined(USE_CUDA_AWARE_MPI)
        CudaMemcpyHtoD(&d_sbuf_[pindex_[owner]*bufsize_], &sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]]*sizeof(GraphElem));
        // CudaDeviceSynchronize(); -- not needed if sbuf_ is pinned
        // https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
        MPI_Isend(&d_sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
#else
        MPI_Isend(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
#endif
        stat_[pindex_[owner]] = '1'; 
      }
    }

    inline void nbsend()
    {
#if defined(USE_OPENMP)
#pragma omp parallel for default(shared)
#endif
      for (int p = 0; p < pdegree_; p++)
        nbsend(targets_[p]);
    }

    inline void nbrecv(int source)
    {
        if (recv_count_[source] > 0)
        {
#if defined(USE_CUDA_AWARE_MPI)
          MPI_Irecv(&d_rbuf_[rindex_[source]*bufsize_], bufsize_, 
              MPI_GRAPH_TYPE, source, TAG_DATA, comm_, &rreq_[rindex_[source]]);
#else
          MPI_Irecv(&rbuf_[rindex_[source]*bufsize_], bufsize_, 
              MPI_GRAPH_TYPE, source, TAG_DATA, comm_, &rreq_[rindex_[source]]);
#endif
        }
    }

    inline void nbrecv()
    {
#if defined(USE_OPENMP)
#pragma omp parallel for default(shared)
#endif
      for (int p = 0; p < rdegree_; p++)
        nbrecv(sources_[p]);
    }
    
    inline void lookup_edges()
    {
      if (pdegree_ == 0)
        return;
      
      const GraphElem lnv = g_->get_lnv();

      for (GraphElem i = 0; i < lnv; i++)
      {
        if (vcount_[i] == 0) // all edges processed, move on
          continue;

        GraphElem e0, e1;
        g_->edge_range(i, e0, e1);
        GraphElem global_i = g_->local_to_global(i);

        if ((e0 + 1) == e1)
          continue;

        for (GraphElem m = e0; m < e1-1; m++)
        {
          EdgeStat& edge = g_->get_edge_stat(m);
          const int owner = g_->get_owner(edge.edge_->tail_);
          
          if (global_i >= edge.edge_->tail_)
              continue;

          const GraphElem pidx = pindex_[owner];
          const GraphElem disp = pidx*bufsize_;

          if (edge.active_ && owner > rank_)
          {   
            if (stat_[pidx] == '1') 
              continue;

            if (m >= prev_m_[pidx])
            {
              if (sbuf_ctr_[pidx] == bufsize_)
              {
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;

                nbsend(owner);

                continue;
              }

              for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
              { 
                Edge const& edge_n = g_->get_edge(n); 
                    
                if (!edge_within_max(edge.edge_->tail_, edge_n.tail_))
                  break;
                
                if (sbuf_ctr_[pidx] == bufsize_)
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = n;

                  nbsend(owner);

                  break;
                }

                if (!edge_above_min(edge.edge_->tail_, edge_n.tail_))
                  continue;

                sbuf_[disp+sbuf_ctr_[pidx]]   = edge.edge_->tail_;
                sbuf_[disp+sbuf_ctr_[pidx]+1] = edge_n.tail_;
                sbuf_ctr_[pidx] += 2;

                out_nghosts_ -= 2;
                vcount_[i] -= 1;
              }

              if (stat_[pidx] == '0') 
              {               
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;
                
                edge.active_ = false;

                if (sbuf_ctr_[pidx] == bufsize_)
                  nbsend(owner);
              }
            }
          }
        }
      }
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
   
    inline void process_recvs()
    {
      if (in_nghosts_ == 0)
        return;

      GraphElem base = g_->get_base(rank_);
#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
      GraphElem nst = 0;
#endif

#if defined(USE_OPENMP)
#pragma omp parallel for default(shared) reduction(-:in_nghosts_)
#endif
      for (GraphElem p = 0; p < rdegree_; p++)
      {
        GraphElem count = 0;
        int over = -1;
        MPI_Status rstat;

        MPI_Test(&rreq_[p], &over, &rstat);

        if (over)
        {
          const int source = rstat.MPI_SOURCE;
          MPI_Get_count(&rstat, MPI_GRAPH_TYPE, (int*)&count);

          if (source != MPI_ANY_SOURCE)
          {
#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
            const int s_idx = nst % count_rstreams_;
            CudaMemcpyAsyncHtoD(&d_rbuf_[s_idx*bufsize_], &rbuf_[p*bufsize_], count*sizeof(GraphElem), e_rstreams_[s_idx]);
            local_count_gpu_recv(count, base, &d_rbuf_[s_idx*bufsize_], d_eind_, d_elst_, d_ntriangles_, e_rstreams_[s_idx]);
            nst++;
#elif defined(USE_CUDA_AWARE_MPI)
            CudaDeviceSynchronize();
            local_count_gpu_recv(count, base, &d_rbuf_[p*bufsize_], d_eind_, d_elst_, d_ntriangles_);
#else
            CudaMemcpyHtoD(d_rbuf_, &rbuf_[p*bufsize_], count*sizeof(GraphElem));
            local_count_gpu_recv(count, base, d_rbuf_, d_eind_, d_elst_, d_ntriangles_);
#endif

            in_nghosts_ -= count;
#if defined(USE_OPENMP)
#pragma omp atomic update
#endif
            recv_count_[source] -= count;

            nbrecv(source);
          }
        }
      }
    }

    inline GraphElem count()
    {
#if defined(USE_ALLREDUCE_FOR_EXIT)
      GraphElem count;
#else      
      bool done = false, nbar_active = false; 
      MPI_Request nbar_req = MPI_REQUEST_NULL;
#endif

      int* inds = new int[pdegree_]();
      int over = -1;
      
      nbrecv();

#if defined(USE_ALLREDUCE_FOR_EXIT)
      while(1)
#else
      while(!done)
#endif
      {
        process_recvs();

        if (out_nghosts_ == 0)
          nbsend();
        else
          lookup_edges();
        
        MPI_Testsome(pdegree_, sreq_, &over, inds, MPI_STATUSES_IGNORE);

        for (int i = 0; i < over; i++)
        {
          sbuf_ctr_[inds[i]] = 0;
          stat_[inds[i]] = '0';
        }

#if defined(USE_ALLREDUCE_FOR_EXIT)
        count = in_nghosts_;
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
        if (count == 0)
          break;
#else       
        if (nbar_active)
        {
          int test_nbar = -1;
          MPI_Test(&nbar_req, &test_nbar, MPI_STATUS_IGNORE);
          done = !test_nbar ? false : true;
        }
        else
        {
          if (in_nghosts_ == 0)
          {
            MPI_Ibarrier(comm_, &nbar_req);
            nbar_active = true;
          }
        }
#endif
#if defined(DEBUG_PRINTF)
        std::cout << "in/out: " << in_nghosts_ << ", " << out_nghosts_ << std::endl;
#endif
      }
   
#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
      for (int t = 0; t < count_rstreams_; t++)
        cudaStreamDestroy(e_rstreams_[t]);
#endif
      CudaDeviceSynchronize();
      
      CudaMemcpyDtoH(&ntriangles_, d_ntriangles_, sizeof(GraphElem));

      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
      
      delete []inds;
      
      return ttc;
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_, rdegree_;
    GraphElem *sbuf_, *recv_count_, *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;

#if defined(USE_CUDA_AWARE_MPI)
    GraphElem *d_sbuf_;
#else
    GraphElem *rbuf_;
#endif

    GraphElem *d_eind_, *d_elst_, *d_rbuf_, *d_ntriangles_; // device
    
    MPI_Request *sreq_, *rreq_;
    char *stat_;
#if defined(USE_ASYNC_STREAMS_AT_RECV) && !defined(USE_CUDA_AWARE_MPI)
    cudaStream_t* e_rstreams_;
    int count_rstreams_;
#endif   
    int rank_, size_;
    std::vector<int> targets_, sources_;
    std::unordered_map<int, int> pindex_, rindex_; 
    
    MPI_Comm comm_;
};
#endif
#endif
