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
#ifndef XXTRIC_HPP
#define XXTRIC_HPP


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

class TriangleCounting
{
  public:

    TriangleCounting(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr), 
      pdegree_(0), sreq_(nullptr), erange_(nullptr), vcount_(nullptr), ntriangles_(0), 
      nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), 
      prev_k_(nullptr), stat_(nullptr), targets_(0), sources_(0), bufsize_(0), rreq_(nullptr),
      recv_count_(nullptr), rindex_(0), rdegree_(0), d_eind_(nullptr), d_elst_(nullptr), 
      d_rbuf_(nullptr), d_ntriangles_(nullptr)
#if defined(USE_ASYNC_STREAMS_AT_RECV)
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

#if defined(LESS_GLOBALMEM_READS)
#else
    GraphElem* d_range = nullptr;
    CudaMalloc(d_range, 2*lnv*sizeof(GraphElem));
#endif
    CudaMalloc(d_eind_, (lnv+1)*sizeof(GraphElem));
    CudaMalloc(d_elst_, lne*sizeof(GraphElem));
    CudaMalloc(d_ntriangles_, sizeof(GraphElem));

    CudaMemcpyHtoD(d_eind_, g_->edge_indices_.data(), (lnv+1)*sizeof(GraphElem));
    CudaMemcpyHtoD(d_elst_, etails.data(), lne*sizeof(GraphElem));
#if defined(LESS_GLOBALMEM_READS)
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

        if (global_i > edge_m.tail_)
          continue;

        if (owner > rank_)
        {
          for (GraphElem n = m + 1; n < e1; n++)
          {
            Edge const& edge_n = g_->get_edge(n);

            if (edge_m.tail_ > edge_n.tail_)
              continue;

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
        
        if (global_i > edge_m.tail_)
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

              if (edge_m.tail_ > edge_n.tail_)
                continue;
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
#else
    /// create streams
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      cudaStreamCreate(&streams[t]);

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
        
        if (global_i > edge_m.tail_)
          continue;
        
        if (owner == rank_)
        {
          GraphElem jstart, jend;
          g_->edge_range(g_->global_to_local(edge_m.tail_), jstart, jend);
          local_count_gpu_inner(m+1, e1, jstart, jend, d_elst_, d_ntriangles_, &streams[i % MAX_CUDA_STREAMS]);
        }
        else
        {
          if (owner > rank_)
          {
            for (GraphElem n = m + 1; n < e1; n++)
            {
              Edge const& edge_n = g_->get_edge(n);

              if (edge_m.tail_ > edge_n.tail_)
                continue;
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
    {
      cudaStreamSynchronize(streams[t]);
      cudaStreamDestroy(streams[t]);
    }
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
 
    sbuf_     = new GraphElem[pdegree_*bufsize_];
    sbuf_ctr_ = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    sreq_     = new MPI_Request[pdegree_];
    stat_     = new char[pdegree_];
    
    rbuf_     = new GraphElem[rdegree_*bufsize_];
    rreq_     = new MPI_Request[rdegree_];

    
    std::fill(sreq_, sreq_ + pdegree_, MPI_REQUEST_NULL);
    std::fill(prev_k_, prev_k_ + pdegree_, -1);
    std::fill(prev_m_, prev_m_ + pdegree_, -1);
    std::fill(stat_, stat_ + pdegree_, '0');
    
    std::fill(rreq_, rreq_ + rdegree_, MPI_REQUEST_NULL);

#if defined(USE_ASYNC_STREAMS_AT_RECV)
    count_rstreams_ = MIN(MAX_CUDA_STREAMS, rdegree_);
    e_rstreams_ = new cudaStream_t[count_rstreams_];
    for (int t = 0; t < count_rstreams_; t++)
      cudaStreamCreate(&e_rstreams_[t]);
    CudaMalloc(d_rbuf_, count_rstreams_*bufsize_*sizeof(GraphElem));
#else
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
#if defined(LESS_GLOBALMEM_READS)
#else
    CudaFree(d_range);
#endif
#if defined(MASKED_LOCAL_COUNTING_KERNEL)
    CudaFree(d_edge_mask);
#endif
  }

    ~TriangleCounting() {}

    void clear()
    {
      delete []sbuf_;
      delete []rbuf_;
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

#if defined(USE_ASYNC_STREAMS_AT_RECV)
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
        MPI_Isend(&sbuf_[pindex_[owner]*bufsize_], sbuf_ctr_[pindex_[owner]], 
            MPI_GRAPH_TYPE, owner, TAG_DATA, comm_, &sreq_[pindex_[owner]]);
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
          MPI_Irecv(&rbuf_[rindex_[source]*bufsize_], bufsize_, 
              MPI_GRAPH_TYPE, source, TAG_DATA, comm_, &rreq_[rindex_[source]]);
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
          
          if (global_i > edge.edge_->tail_)
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
                    
                if (edge.edge_->tail_ > edge_n.tail_)
                  continue;

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
#if defined(USE_ASYNC_STREAMS_AT_RECV)
            const int s_idx = p % count_rstreams_;
            CudaMemcpyAsyncHtoD(&d_rbuf_[s_idx*bufsize_], &rbuf_[p*bufsize_], count*sizeof(GraphElem), e_rstreams_[s_idx]);
            local_count_gpu_recv(count, base, &d_rbuf_[s_idx*bufsize_], d_eind_, d_elst_, d_ntriangles_, e_rstreams_[s_idx]);
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
   
#if defined(USE_ASYNC_STREAMS_AT_RECV)
      for (int t = 0; t < count_rstreams_; t++)
      {
        cudaStreamSynchronize(e_rstreams_[t]);
        cudaStreamDestroy(e_rstreams_[t]);
      }
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
    GraphElem *sbuf_, *rbuf_, *recv_count_, *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;

    GraphElem *d_eind_, *d_elst_, *d_rbuf_, *d_ntriangles_; // device
    
    MPI_Request *sreq_, *rreq_;
    char *stat_;
#if defined(USE_ASYNC_STREAMS_AT_RECV)
    cudaStream_t* e_rstreams_;
    int count_rstreams_;
#endif   
    std::vector<int> targets_, sources_;
    int rank_, size_;
    std::unordered_map<int, int> pindex_, rindex_; 
    
    MPI_Comm comm_;
};
#endif
