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
#ifndef XXTRICRMA_HPP
#define XXTRICRMA_HPP

#include "graph.hpp"

#include <numeric>
#include <utility>
#include <cstring>
#include <iomanip>
#include <limits>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string> 

class TriangleCounting
{
  public:

    TriangleCounting(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), rbuf_(nullptr),
      sig_addr_(0), pdegree_(0), erange_(nullptr), vcount_(nullptr), ntriangles_(0), 
      nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), 
      prev_k_(nullptr), stat_(nullptr), targets_(0), sources_(0), bufsize_(0), tdisp_(nullptr),
      recv_count_(nullptr), rindex_(0), rdegree_(0)
  {
    comm_ = g_->get_comm();
    rank_ = shmem_my_pe();
    size_ = shmem_n_pes();

    const GraphElem lnv = g_->get_lnv();
    const GraphElem nv = g_->get_nv();

    vcount_ = new GraphElem[lnv]();
    erange_ = new GraphElem[nv*2]();
    
    GraphElem *send_count  = (GraphElem *)shmem_malloc(size_ * sizeof(GraphElem));
    recv_count_  = (GraphElem *)shmem_malloc(size_ * sizeof(GraphElem));
    std::memset(send_count, 0, size_*sizeof(GraphElem));
    std::memset(recv_count_, 0, size_*sizeof(GraphElem));

    double t0 = MPI_Wtime();

    // store edge ranges
    GraphElem base = g_->get_base(rank_);
    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);

      if ((e0 + 1) == e1)
        continue;

      Edge const& edge_s = g_->get_edge(e0);
      Edge const& edge_t = g_->get_edge(e1-1);

      erange_[(i + base)*2] = edge_s.tail_;
      erange_[(i + base)*2+1] = edge_t.tail_;
    }
    
    MPI_Barrier(comm_);
    
    MPI_Allreduce(MPI_IN_PLACE, erange_, nv*2, MPI_GRAPH_TYPE, 
        MPI_SUM, comm_);

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
        tup[0] = edge_m.tail_;

        if (global_i > tup[0])
            continue;

        if (owner == rank_)
        {
            for (GraphElem n = m + 1; n < e1; n++)
            {
                Edge const& edge_n = g_->get_edge(n);

                tup[1] = edge_n.tail_;
#if defined(FAST_CHECK_EDGELIST)
                if (fast_check_edgelist(tup))
#else
                if (check_edgelist(tup))
#endif
                    ntriangles_ += 1;
            }
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

                    send_count[owner] += 1;
                    vcount_[i] += 1;
                }
            }
        }
      }
    }

    MPI_Barrier(comm_);
    shmem_team_sync(SHMEM_TEAM_WORLD);
        
    double t1 = MPI_Wtime();
    double p_tot = t1 - t0, t_tot = 0.0;

    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

    if (rank_ == 0) 
    {   
      std::cout << "Average time for local counting during instantiation (secs.): " 
        << ((double)(t_tot / (double)size_)) << std::endl;
    }

    t0 = MPI_Wtime();

    // outgoing/incoming data and buffer size
    shmem_int64_alltoall(SHMEM_TEAM_WORLD, recv_count_, send_count, 1);
    
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

    bufsize_ = ((nghosts_*2) < bufsize) ? (nghosts_*2) : bufsize;

    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

    if (rank_ == 0)
      std::cout << "Adjusted Per-PE buffer count: " << bufsize_ << std::endl;

    rbuf_ = (GraphElem *)shmem_malloc(rdegree_*bufsize_*sizeof(GraphElem));
    std::fill(rbuf_, rbuf_ + rdegree_*bufsize_, -999);
    
    sbuf_ctr_ = new GraphElem[pdegree_]();
    prev_k_   = new GraphElem[pdegree_];
    prev_m_   = new GraphElem[pdegree_];
    stat_     = new char[pdegree_];
    sbuf_     = new GraphElem[pdegree_*bufsize_];
    
    tdisp_    = (GraphElem *)shmem_malloc(size_*sizeof(GraphElem));

    std::fill(prev_k_, prev_k_ + pdegree_, -1);
    std::fill(prev_m_, prev_m_ + pdegree_, -1);
    std::fill(stat_, stat_ + pdegree_, '0');

    // TODO FIXME use neighbor collectives instead 
    std::fill(send_count, send_count + size_, 0);
    GraphElem disp = 0;
    
    for (int i = 0; i < rdegree_; i++)
    {
        send_count[sources_[i]] = disp;
        disp += bufsize_;
    }
    
    MPI_Barrier(comm_);
    shmem_team_sync(SHMEM_TEAM_WORLD);
   
    shmem_int64_alltoall(SHMEM_TEAM_WORLD, tdisp_, send_count, 1);
    
#if defined(DEBUG_PRINTF)
    if (rank_ == 0)
    {
      std::cout << "Edge range per vertex (#ID: <range>): " << std::endl;
      for (int i = 0, j = 0; i < nv*2; i+=2, j++)
        std::cout << j << ": " << erange_[i] << ", " << erange_[i+1] << std::endl;
    }
#endif
    
     shmem_free(send_count);
  }

    ~TriangleCounting() {}

    void clear()
    {
      delete []sbuf_;
      delete []sbuf_ctr_;
      delete []prev_k_;
      delete []prev_m_;
      delete []stat_;
      delete []vcount_;
      delete []erange_;
      
      shmem_free(recv_count_);
      shmem_free(rbuf_);
      shmem_free(tdisp_);

      pindex_.clear();
      targets_.clear();
      rindex_.clear();
      sources_.clear();
    }

    inline void rput(int owner)
    {
      if (stat_[pindex_[owner]] == '0' && sbuf_ctr_[pindex_[owner]] > 0)
      {
        shmem_int64_put_signal_nbi(&rbuf_[tdisp_[owner]], &sbuf_[pindex_[owner]*bufsize_], 
            sbuf_ctr_[pindex_[owner]], &sig_addr_, 1, SHMEM_SIGNAL_ADD, owner);    
        stat_[pindex_[owner]] = '1'; 
      }
    }

    inline void rput()
    {
#if defined(USE_OPENMP)
#pragma omp parallel for default(shared)
#endif
      for (int p = 0; p < pdegree_; p++)
        rput(targets_[p]);
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
              if (sbuf_ctr_[pidx] == (bufsize_-1))
              {
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;

                rput(owner);

                continue;
              }

              sbuf_[disp+sbuf_ctr_[pidx]] = edge.edge_->tail_;
              sbuf_ctr_[pidx] += 1;

              for (GraphElem n = ((prev_k_[pidx] == -1) ? (m + 1) : prev_k_[pidx]); n < e1; n++)
              {  
                Edge const& edge_n = g_->get_edge(n); 

                if (!edge_within_max(edge.edge_->tail_, edge_n.tail_))
                  break;

                if (sbuf_ctr_[pidx] == (bufsize_-1))
                {
                  prev_m_[pidx] = m;
                  prev_k_[pidx] = n;

                  sbuf_[disp+sbuf_ctr_[pidx]] = -1; // demarcate vertex boundary
                  sbuf_ctr_[pidx] += 1;

                  rput(owner);

                  break;
                }

                if (!edge_above_min(edge.edge_->tail_, edge_n.tail_))
                  continue;

                sbuf_[disp+sbuf_ctr_[pidx]] = edge_n.tail_;
                sbuf_ctr_[pidx] += 1;

                out_nghosts_ -= 1;
                vcount_[i] -= 1;
              }

              if (stat_[pidx] == '0') 
              {               
                prev_m_[pidx] = m;
                prev_k_[pidx] = -1;
                
                edge.active_ = false;

                if (sbuf_ctr_[pidx] == (bufsize_-1))
                {
                  sbuf_[disp+sbuf_ctr_[pidx]] = -1; 
                  sbuf_ctr_[pidx] += 1;

                  rput(owner);
                }
                else
                {
                  sbuf_[disp+sbuf_ctr_[pidx]] = -1; 
                  sbuf_ctr_[pidx] += 1;
                }
              }
            }
          }
        }
      }
    }

    inline bool check_edgelist(GraphElem (&tup)[2])
    {
      GraphElem e0, e1;
      const GraphElem lv = g_->global_to_local(tup[0]);
      g_->edge_range(lv, e0, e1);
      for (GraphElem e = e0; e < e1; e++)
      {
        Edge const& edge = g_->get_edge(e);
        if (edge.tail_ > tup[1])
          break;
        if (tup[1] == edge.tail_)
          return true;
      }
      return false;
    }
    
    inline bool fast_check_edgelist(GraphElem (&tup)[2])
    {
      GraphElem e0, e1;
      const GraphElem lv = g_->global_to_local(tup[0]);
      
      g_->edge_range(lv, e0, e1);
      Edge const& first_edge = g_->get_edge(e0);
      Edge const& last_edge = g_->get_edge(e1 - 1);

      if (tup[1] == last_edge.tail_ || tup[1] == first_edge.tail_)
        return true;
      
      if (tup[1] > first_edge.tail_ && tup[1] < last_edge.tail_)
      {
        Edge const& mid_edge = g_->get_edge(std::midpoint(e0, e1));

        if (tup[1] == mid_edge.tail_)
          return true;
        else if (tup[1] > mid_edge.tail_)
        {
          for (GraphElem e = std::midpoint(e0, e1) + 1; e < e1; e++)
          {
            Edge const& edge = g_->get_edge(e);
            if (edge.tail_ > tup[1])
              break;
            if (tup[1] == edge.tail_)
              return true;
          }
        }
        else
        {
          for (GraphElem e = e0; e < std::midpoint(e0, e1); e++)
          {
            Edge const& edge = g_->get_edge(e);
            if (edge.tail_ > tup[1])
              break;
            if (tup[1] == edge.tail_)
              return true;
          }
        }
      }
        
      return false;
    }

#if defined(USE_OPENMP_NESTED)
    inline bool check_edgelist_omp(GraphElem (&tup)[2])
    {
      GraphElem e0, e1;
      bool found = false;
      
      const GraphElem lv = g_->global_to_local(tup[0]);
      g_->edge_range(lv, e0, e1);

#pragma omp parallel for schedule(dynamic) firstprivate(tup) shared(g_, e0, e1) reduction(||:found) if (e1 - e0 >= 50)
      for (GraphElem e = e0; e < e1; e++)
      {
        Edge const& edge = g_->get_edge(e);
        
        if (tup[1] == edge.tail_)
          found = found || true;
      }

      return found;
    }
#endif

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

    inline void process_window()
    {
      if (in_nghosts_ == 0)
        return;

      if (shmem_uint64_test(&sig_addr_, SHMEM_CMP_GE, 1))
      {
#if defined(USE_OPENMP)
#pragma omp parallel for default(shared) reduction(+:ntriangles_) reduction(-:in_nghosts_)
#else
#endif
        for (int p = 0; p < rdegree_; p++)
        {
          GraphElem tup[2] = {-1,-1}, prev = 0, k = 0;
          const int source = sources_[p];

          if ((rbuf_[p*bufsize_] != -999) && (recv_count_[source] > 0))
          {
            bool done = false;

            while (!done)
            {
              if (k >= bufsize_)
                break;

              if (rbuf_[p*bufsize_+k] == -999)
                break;

              if (rbuf_[p*bufsize_+k] == -1)
              {
                k += 1;
                prev = k;
                continue;
              }

              tup[0] = rbuf_[p*bufsize_+k];
              GraphElem curr_count = 0;

              for (GraphElem m = k + 1; m < bufsize_; m++)
              {
                if (rbuf_[p*bufsize_+m] == -999)
                {
                  done = true;
                  break;
                }

                if (rbuf_[p*bufsize_+m] == -1)
                {
                  curr_count = m + 1;
                  break;
                }

                tup[1] = rbuf_[p*bufsize_+m];

#if defined(USE_OPENMP) && defined(USE_OPENMP_NESTED)
                if (check_edgelist_omp(tup))
#elif defined(FAST_CHECK_EDGELIST)
                  if (fast_check_edgelist(tup))
#else
                    if (check_edgelist(tup))
#endif
                      ntriangles_ += 1;

                in_nghosts_ -= 1;
                recv_count_[source] -= 1;
              }

              k += (curr_count - prev);
              prev = k;
            }

            // reset window buffer
            if (recv_count_[source] > 0)
              std::fill(rbuf_ + p*bufsize_, rbuf_ + p*bufsize_ + bufsize_, -999);
          }
        }
        sig_addr_ = 0;
      }
    }

    inline GraphElem count()
    {
      GraphElem count;

      while(1)
      {
        if (out_nghosts_ == 0)
          rput();
        else
          lookup_edges();

        shmem_quiet();

        for (GraphElem i = 0; i < pdegree_; i++)
        {
          if (stat_[i] == '1')
          {
            sbuf_ctr_[i] = 0;
            stat_[i] = '0';
          }
        }

        process_window();

        shmem_int64_sum_reduce(SHMEM_TEAM_WORLD, &count, &in_nghosts_, 1);

        if (count == 0)
          break;

#if defined(DEBUG_PRINTF)
        std::cout << "in/out: " << in_nghosts_ << ", " << out_nghosts_ << std::endl;
#endif
      }

      GraphElem ttc = 0, ltc = ntriangles_;
      MPI_Barrier(comm_);
      MPI_Reduce(&ltc, &ttc, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
      
      return ttc;
    }

  private:
    Graph* g_;

    GraphElem ntriangles_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_, rdegree_;
    uint64_t sig_addr_;
    GraphElem *sbuf_, *rbuf_, *tdisp_, *recv_count_, *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;
    char *stat_;
    
    std::vector<int> targets_, sources_;
    int rank_, size_;
    std::unordered_map<int, int> pindex_, rindex_; 

    MPI_Comm comm_;
};
#endif
