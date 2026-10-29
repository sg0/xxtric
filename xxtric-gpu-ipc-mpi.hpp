#pragma once
#ifndef XXTRIC_GPU_IPC_MPI_HPP
#define XXTRIC_GPU_IPC_MPI_HPP

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

/// MPI and IPC based distributed-memory counting
class TriangleCounting
{
  public:

    TriangleCounting(Graph* g, const GraphElem bufsize): 
      g_(g), sbuf_ctr_(nullptr), sbuf_(nullptr), pdegree_(0), 
      sreq_(nullptr), erange_(nullptr), vcount_(nullptr), ntriangles_(0), 
      nghosts_(0), out_nghosts_(0), in_nghosts_(0), pindex_(0), prev_m_(nullptr), 
      prev_k_(nullptr), stat_(nullptr), targets_(0), sources_(0), tstatus_(0), bufsize_(0), rreq_(nullptr),
      recv_count_(nullptr), rindex_(0), rdegree_(0), d_eind_(nullptr), d_elst_(nullptr), 
      d_rbuf_(nullptr), d_ntriangles_(nullptr), rbuf_(nullptr), e_rstreams_(nullptr), count_rstreams_(0),
      shm_comm_(MPI_COMM_NULL)
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
    // TODO FIXME this test can be expensive due to O(2*nv) memory
    const GraphElem base = g_->get_base(rank_);

    for (GraphElem i = 0; i < lnv; i++)
    {
      GraphElem e0, e1;
      g_->edge_range(i, e0, e1);
      const GraphElem global_i = g_->local_to_global(i);

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
   
    int shm_size = 0, shm_rank = MPI_UNDEFINED;
    MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_, 
        MPI_INFO_NULL, &shm_comm_);
    MPI_Comm_size(shm_comm_, &shm_size);
    MPI_Comm_rank(shm_comm_, &shm_rank);

    /// assume all tasks/node can see all GPUs in a node
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (rank_ == 0 && (deviceCount != shm_size)) 
    {
      std::cerr << "#devices does not match #processes in local subgroup." << std::endl;
      MPI_Abort(comm_, -99);
    }

    cudaSetDevice(shm_rank);

    CudaMalloc(d_eind_, (lnv+1)*sizeof(GraphElem));
    CudaMalloc(d_elst_, lne*sizeof(GraphElem));
    CudaMalloc(d_ntriangles_, sizeof(GraphElem));

    CudaMemcpyHtoD(d_eind_, g_->edge_indices_.data(), (lnv+1)*sizeof(GraphElem));
    CudaMemcpyHtoD(d_elst_, etails.data(), lne*sizeof(GraphElem));
    
    CudaMemset(d_ntriangles_, 0, sizeof(GraphElem));

    GraphElem nst = 0;
    cudaStream_t* streams = new cudaStream_t[MAX_CUDA_STREAMS];
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
      CudaStreamCreate(streams[t]);
       
    cudaIpcMemHandle_t local_elst_handle, local_eind_handle;
    cudaIpcMemHandle_t *d_eind_handles = new cudaIpcMemHandle_t[shm_size];
    cudaIpcMemHandle_t *d_elst_handles = new cudaIpcMemHandle_t[shm_size];
    
    int *shm_nbrs = new int[shm_size];
    MPI_Allgather(&rank_, 1, MPI_INT, shm_nbrs, 1, MPI_INT, shm_comm_);
   
    CudaIPCGetMemHandle(d_eind_, local_eind_handle);
    CudaIPCGetMemHandle(d_elst_, local_elst_handle);

    MPI_Allgather(&local_eind_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
        d_eind_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, shm_comm_);
    MPI_Allgather(&local_elst_handle, sizeof(cudaIpcMemHandle_t), MPI_BYTE, 
        d_elst_handles, sizeof(cudaIpcMemHandle_t), MPI_BYTE, shm_comm_);
   
    std::vector<GraphElem*> d_eind_nbr(shm_size, nullptr), d_elst_nbr(shm_size, nullptr);

    GraphElem** d_edge_mask_nbr = new GraphElem*[shm_size];
    std::vector<std::vector<GraphElem>> v_edge_mask_nbr(shm_size);
    
    GraphElem* d_edge_mask = nullptr;
    std::vector<GraphElem> v_edge_mask;

    for (int t = 0; t < shm_size; t++)
    {    
      CudaIPCOpenMemHandle(d_eind_nbr[t], d_eind_handles[t]);
      CudaIPCOpenMemHandle(d_elst_nbr[t], d_elst_handles[t]);
    }

    MPI_Barrier(comm_);

    /// Counting buffers preparation
    
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

        if (owner == rank_)
        {
          v_edge_mask.push_back(e1);
          v_edge_mask.push_back(m);
        }
        else
        {
          if (owner > rank_)
          {
            int *res = std::find(shm_nbrs, shm_nbrs + shm_size, owner);

            if (res != (shm_nbrs + shm_size))
            {
              const GraphElem base_m = edge_m.tail_-g_->get_base(owner);           
              const GraphElem idx = (GraphElem)(res - shm_nbrs);
              
              v_edge_mask_nbr[idx].push_back(e1);
              v_edge_mask_nbr[idx].push_back(m);
              v_edge_mask_nbr[idx].push_back(base_m);
               
              if (send_count[owner] != -1) 
                send_count[owner] = -1;
            }
            else
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
    }

    GraphElem local_nedges_rem = 0;

    /// within the same node 
    for (int p = 0; p < shm_size; p++)
    {
      if (!v_edge_mask_nbr[p].empty())
      {
        const GraphElem nedges_rem = v_edge_mask_nbr[p].size()/3;
        CudaMalloc(d_edge_mask_nbr[p], nedges_rem*3*sizeof(GraphElem));
        CudaMemcpyHtoD(d_edge_mask_nbr[p], v_edge_mask_nbr[p].data(), nedges_rem*3*sizeof(GraphElem));

        local_count_gpu_masked_ext(nedges_rem, d_edge_mask_nbr[p], (GraphElem*)d_eind_nbr[p], (GraphElem*)d_elst_nbr[p], d_elst_, d_ntriangles_, streams[nst % MAX_CUDA_STREAMS]);
        nst++;
        local_nedges_rem += nedges_rem;
      }
    }

    /// local
    const GraphElem nedges = v_edge_mask.size()/2;
    CudaMalloc(d_edge_mask, nedges*2*sizeof(GraphElem));
    CudaMemcpyHtoD(d_edge_mask, v_edge_mask.data(), nedges*2*sizeof(GraphElem));
   
    // grid-block over vertices
    local_count_gpu_masked(nedges, base, d_edge_mask, d_eind_, d_elst_, d_ntriangles_, streams[nst % MAX_CUDA_STREAMS]);
   
    for (int t = 0; t < shm_size; t++)
    {    
      CudaIPCCloseMemHandle(d_eind_nbr[t]);
      CudaIPCCloseMemHandle(d_elst_nbr[t]);

      v_edge_mask_nbr[t].clear();
    }
        
    d_eind_nbr.clear();
    d_elst_nbr.clear();

    v_edge_mask_nbr.clear();
    v_edge_mask.clear();
    
    for (int t = 0; t < MAX_CUDA_STREAMS; t++)
    {
      CudaStreamSynchronize(streams[t]);
      CudaStreamDestroy(streams[t]);
    }
    
    CudaDeviceSynchronize();

    double t1 = MPI_Wtime(); 
    double p_tot = t1 - t0, t_tot = 0.0, t_max = 0.0, t_min = 0.0;
    GraphElem sum_nedges_rem = 0, min_nedges_rem = 0, max_nedges_rem = 0;

    MPI_Barrier(comm_);
    
    MPI_Reduce(&p_tot, &t_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
    MPI_Reduce(&p_tot, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
    MPI_Reduce(&p_tot, &t_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm_);
    
    MPI_Reduce(&local_nedges_rem, &sum_nedges_rem, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
    MPI_Reduce(&local_nedges_rem, &max_nedges_rem, 1, MPI_GRAPH_TYPE, MPI_MAX, 0, comm_);
    MPI_Reduce(&local_nedges_rem, &min_nedges_rem, 1, MPI_GRAPH_TYPE, MPI_MIN, 0, comm_);
    
    if (rank_ == 0) 
    {   
      std::cout << "Avg/Min/Max times for local counting during instantiation (secs.): " 
        << ((double)(t_tot / (double)size_)) << ", " << t_min << ", " << t_max << std::endl;
      std::cout << "Avg/Min/Max node-local #remote vertices processed via IPC: " 
        << (GraphElem)(sum_nedges_rem / size_) << ", " << min_nedges_rem << ", " << max_nedges_rem << std::endl;
    }

    t0 = MPI_Wtime();

    // outgoing/incoming data and buffer size
    MPI_Alltoall(send_count, 1, MPI_GRAPH_TYPE, recv_count_, 1, MPI_GRAPH_TYPE, comm_);
    
    for (GraphElem p = 0; p < size_; p++)
    {
      if (send_count[p] > 0)
      {
        out_nghosts_ += send_count[p];
        targets_.push_back(p);
        tstatus_.insert({p, true});
      }

      if (send_count[p] == -1)
        tstatus_.insert({p, false});

      if (recv_count_[p] > 0)
      {
        sources_.push_back(p);
        in_nghosts_ += recv_count_[p];
      }
    }

    pdegree_ = targets_.size();
    rdegree_ = sources_.size();

    for (int i = 0; i < pdegree_; i++)
      pindex_.insert({targets_[i], i});

    for (int i = 0; i < rdegree_; i++)
      rindex_.insert({sources_[i], i});

    nghosts_ = out_nghosts_ + in_nghosts_;

    bufsize_ = ((nghosts_*2 < bufsize) || (bufsize == -1)) ? nghosts_*2 : bufsize;

    MPI_Allreduce(MPI_IN_PLACE, &bufsize_, 1, MPI_GRAPH_TYPE, MPI_MAX, comm_);

#if defined(CUDA_IPC_MPI_DEBUG_STRING)
    std::cout << "Shared comm size: " << shm_size << std::endl;
    for (int i = 0; i < shm_size; i++)
    {
      std::cout << "PE in shared memory node: " << shm_nbrs[i] << std::endl;
    }
    for (int i = 0; i < pdegree_; i++)
    {
      if (!tstatus_[targets_[i]])
        std::cout << "PE in shared comm: " << targets_[i] << std::endl;
    }
#endif

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

    sbuf_     = new GraphElem[pdegree_*bufsize_];
    rbuf_     = new GraphElem[rdegree_*bufsize_];
    count_rstreams_ = MIN(MAX_CUDA_STREAMS, rdegree_);
    e_rstreams_ = new cudaStream_t[count_rstreams_];
    for (int t = 0; t < count_rstreams_; t++)
      cudaStreamCreate(&e_rstreams_[t]);
    CudaMalloc(d_rbuf_, count_rstreams_*bufsize_*sizeof(GraphElem));

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
    delete []shm_nbrs;

    CudaFree(d_edge_mask);
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

      delete []e_rstreams_;
      
      pindex_.clear();
      targets_.clear();
      rindex_.clear();
      tstatus_.clear();
      sources_.clear();

      MPI_Comm_free(&shm_comm_);
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

          if (owner > rank_ && tstatus_[owner])
          {
            const GraphElem pidx = pindex_[owner];
            const GraphElem disp = pidx*bufsize_;

            if (edge.active_)
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
      GraphElem nst = 0;

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
            const int s_idx = nst % count_rstreams_;
            CudaMemcpyAsyncHtoD(&d_rbuf_[s_idx*bufsize_], &rbuf_[p*bufsize_], count*sizeof(GraphElem), e_rstreams_[s_idx]);
            local_count_gpu_recv(count, base, &d_rbuf_[s_idx*bufsize_], d_eind_, d_elst_, d_ntriangles_, e_rstreams_[s_idx]);
            nst++;

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
   
      for (int t = 0; t < count_rstreams_; t++)
        cudaStreamDestroy(e_rstreams_[t]);
      
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

    GraphElem ntriangles_, bufsize_, nghosts_, out_nghosts_, in_nghosts_, pdegree_, rdegree_, count_rstreams_;
    GraphElem *sbuf_, *recv_count_, *prev_k_, *prev_m_, *sbuf_ctr_, *vcount_, *erange_;

    GraphElem *rbuf_;

    GraphElem *d_eind_, *d_elst_, *d_rbuf_, *d_ntriangles_; // device
    
    MPI_Request *sreq_, *rreq_;
    char *stat_;
    cudaStream_t* e_rstreams_;
    int rank_, size_;
    std::vector<int> targets_, sources_;
    std::unordered_map<int, int> pindex_, rindex_;
    std::unordered_map<int, bool> tstatus_;
    
    MPI_Comm comm_, shm_comm_;
};
#endif
