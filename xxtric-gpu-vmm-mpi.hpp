#pragma once
#ifndef XXTRIC_GPU_VMM_MPI_HPP
#define XXTRIC_GPU_VMM_MPI_HPP

// Compile-time options
//   XXTRIC_VMM_NO_ERANGE_FILTER   skip the min/max-neighbour pruning (saves the
//                                 O(2*nv) replicated array, sends more pairs)
//   XXTRIC_VMM_NO_GDR             do not request GPUDirect-RDMA-capable memory
//
// Run-time environment
//   XXTRIC_VMM_COMM_MB            comm buffer budget per GPU in MB, or
//   XXTRIC_VMM_COMM_FRAC          fraction of free GPU memory (default 0.5);

#include <mpi.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vmm_mpi.h"         // vmm::DistributedBuffer (includes vmm_multigpu.h)
#include "counting.cuh"  

#ifndef XXTRIC_VMM_TAG_BASE
#define XXTRIC_VMM_TAG_BASE (100)
#endif

class TriangleCounting
{
  public:
    explicit TriangleCounting(Graph* g, const GraphElem bufsize = -1)
      : g_(g), comm_(g->get_comm())
    {
      MPI_Comm_size(comm_, &size_);
      MPI_Comm_rank(comm_, &rank_);
      try {
        setup(bufsize);
      } catch (const std::exception& e) {
        std::cerr << "[rank " << rank_ << "] TriangleCounting (VMM) setup failed: "
                  << e.what() << std::endl;
        MPI_Abort(comm_, -99);
      }
    }

    ~TriangleCounting() { clear(); }

    TriangleCounting(const TriangleCounting&) = delete;
    TriangleCounting& operator=(const TriangleCounting&) = delete;

    /// Must be called before MPI_Finalize (main.cpp does).
    void clear()
    {
      if (cleared_) return;
      cleared_ = true;
      for (auto& G : gpus_) {
        cudaSetDevice(G.dev);
        cudaDeviceSynchronize();
        vmm_plan_free(G.plan);
        cudaFree(G.d_count);
        cudaFree(G.d_parts);
        cudaFree(G.d_slots);
        cudaFree(G.d_tab);
        if (G.ev_gen) cudaEventDestroy(G.ev_gen);
        for (auto& ev : G.ev_rcv) if (ev) cudaEventDestroy(ev);
        if (G.s_cmp) cudaStreamDestroy(G.s_cmp);
        if (G.s_gen) cudaStreamDestroy(G.s_gen);
        if (G.s_rcv) cudaStreamDestroy(G.s_rcv);
      }
      gpus_.clear();
      ncsr_.clear();
      outs_.clear();
      ins_.clear();
      comm_buf_.reset();  // frees the VMM comm chunks; MPI_Comm_free inside is collective
      erange_.reset();
      csr_.reset();
    }

    /// Returns the global count on rank 0 (as the original).
    GraphElem count()
    {
      const double t0 = MPI_Wtime();

      // 1. Node-local wedges: all GPUs, concurrently with the exchange below.
      for (int h = 0; h < ngpus_; h++) {
        Gpu& G = gpus_[h];
        CudaCall(cudaSetDevice(G.dev));
        CudaCall(cudaMemsetAsync(G.d_count, 0, sizeof(VmmCount), G.s_cmp));
        CudaCall(cudaStreamSynchronize(G.s_cmp));  // s_rcv also adds into d_count
        vmm_count_intranode(ncsr_[h], h, G.ne, G.d_count, G.s_cmp);
      }

      // 2. Inter-node wedges: GPU-generated pairs, GPU-to-GPU transfers.
      if (comm_buf_ && max_rounds_ > 0)
        exchange_and_count();

      // 3. Gather.
      VmmCount local = 0;
      for (auto& G : gpus_) {
        CudaCall(cudaSetDevice(G.dev));
        CudaCall(cudaDeviceSynchronize());
        VmmCount c = 0;
        CudaCall(cudaMemcpy(&c, G.d_count, sizeof(c), cudaMemcpyDeviceToHost));
        local += c;
      }
      const double t1 = MPI_Wtime();

      unsigned long long ltc = local, ttc = 0;
      MPI_Reduce(&ltc, &ttc, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm_);

      double tl = t1 - t0, tmax = 0.0;
      MPI_Reduce(&tl, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
      if (rank_ == 0)
        std::cout << "VMM counting: max per-node time (secs.): " << tmax
                  << ", rounds (max over nodes): " << global_rounds_ << std::endl;

      ntriangles_ = (GraphElem)ttc;
      return ntriangles_;
    }

  private:
    // ------------------------------------------------------------------ types
    struct Gpu
    {
      int dev = -1;
      GraphElem nv = 0, ne = 0;
      cudaStream_t s_cmp = nullptr, s_gen = nullptr, s_rcv = nullptr;
      cudaEvent_t ev_gen = nullptr;
      cudaEvent_t ev_rcv[2] = {nullptr, nullptr};
      VmmCount* d_count = nullptr;
      GraphElem* d_parts = nullptr;
      void* d_tab = nullptr;  // this GPU's copy of the NodeCSR tables
      VmmRemotePlan plan;
      std::vector<GraphElem> dbeg, doff, dtot;  // host copies of the plan's dest table
      VmmSendSlot* d_slots = nullptr;
      int nslots = 0;
      GraphElem max_cap = 0;
      size_t comm_bytes = 0;
    };

    struct OutSlot { int g, dest; GraphElem tot, cap; int nsets, rounds; size_t off; };
    struct InSlot  { int g, src, srcgpu; GraphElem tot, cap; int nsets, rounds; size_t off; };

    static constexpr size_t kAlign = 256;
#if defined(XXTRIC_VMM_NO_GDR)
    static constexpr bool kGDR = false;
#else
    static constexpr bool kGDR = true;
#endif

    static size_t align_up(size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }
    static GraphElem round_len(GraphElem tot, GraphElem cap, int k)
    {
      const GraphElem start = (GraphElem)k * cap;
      return start >= tot ? 0 : std::min(cap, tot - start);
    }
    int tag_of(int srcgpu) const { return XXTRIC_VMM_TAG_BASE + srcgpu; }
    int chunk_of(int h) const { return comm_buf_->firstLocalChunk() + h; }

    // ------------------------------------------------------------------ setup
    void setup(const GraphElem bufsize)
    {
      double t0 = MPI_Wtime();

      // One rank per node (fail early, before allocating anything).
      bool one_per_node = true;
      {
        MPI_Comm shm;
        int nloc = 0;
        MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_, MPI_INFO_NULL, &shm);
        MPI_Comm_size(shm, &nloc);
        MPI_Comm_free(&shm);
        int bad = (nloc > 1), anybad = 0;
        MPI_Allreduce(&bad, &anybad, 1, MPI_INT, MPI_LOR, comm_);

        if (anybad) {
          if (rank_ == 0)
            std::cerr << "*Error*: the VMM version expects ONE MPI process per node "
                         "(e.g. mpirun --map-by ppr:1:node, srun --ntasks-per-node=1)."
                      << std::endl;
          MPI_Abort(comm_, -99);
        }
      }

      // Devices of this node.
      CudaCall(cudaGetDeviceCount(&ngpus_));
      if (ngpus_ < 1) throw std::runtime_error("no CUDA devices visible");
      devs_.resize(ngpus_);
      std::iota(devs_.begin(), devs_.end(), 0);
      MPI_Allreduce(&ngpus_, &maxg_, 1, MPI_INT, MPI_MAX, comm_);  // Alltoall stride
      gpus_.resize(ngpus_);

      // 1. Sub-partition the rank's vertices across GPUs, balanced by #edges.
      const GraphElem lnv = g_->get_lnv(), lne = g_->get_lne();
      const std::vector<GraphElem>& ei = g_->edge_indices_;
      const GraphElem nbase = g_->get_base(rank_);
      std::vector<GraphElem> vsl(ngpus_ + 1, 0);
      for (int h = 1; h < ngpus_; h++) {
        const GraphElem target = (lne * h) / ngpus_;
        GraphElem l = std::lower_bound(ei.begin(), ei.begin() + lnv + 1, target) - ei.begin();
        vsl[h] = std::min(std::max(l, vsl[h - 1]), lnv);
      }
      vsl[ngpus_] = lnv;

      // 2. CSR slices in one VMM range (chunk h on GPU h, mapped on all GPUs).
      size_t max_elems = 1;
      for (int h = 0; h < ngpus_; h++) {
        gpus_[h].dev = devs_[h];
        gpus_[h].nv  = vsl[h + 1] - vsl[h];
        gpus_[h].ne  = ei[vsl[h + 1]] - ei[vsl[h]];
        max_elems = std::max(max_elems, (size_t)(gpus_[h].nv + 1 + gpus_[h].ne));
      }
      {
        mem_report("CSR", max_elems * sizeof(GraphElem));
        vmm::MultiGpuBufferOptions o;
        o.bytesPerDevice        = max_elems * sizeof(GraphElem);
        o.devices               = devs_;
        o.requireFullPeerAccess = true;
        csr_.emplace(o);
      }
      std::vector<GraphElem> h_vsplit(ngpus_ + 1);
      std::vector<const GraphElem*> h_ind(ngpus_), h_lst(ngpus_);
      for (int h = 0; h <= ngpus_; h++) h_vsplit[h] = nbase + vsl[h];
      {
        std::vector<GraphElem> buf;
        for (int h = 0; h < ngpus_; h++) {
          Gpu& G = gpus_[h];
          const GraphElem v0 = vsl[h], e0 = ei[v0];
          buf.resize(G.nv + 1 + G.ne);
          for (GraphElem j = 0; j <= G.nv; j++) buf[j] = ei[v0 + j] - e0;
          for (GraphElem e = 0; e < G.ne; e++) buf[G.nv + 1 + e] = g_->get_edge(e0 + e).tail_;
          GraphElem* base = static_cast<GraphElem*>(csr_->chunk(h));
          CudaCall(cudaSetDevice(G.dev));
          CudaCall(cudaMemcpy(base, buf.data(), buf.size() * sizeof(GraphElem),
                             cudaMemcpyHostToDevice));
          h_ind[h] = base;
          h_lst[h] = base + G.nv + 1;
        }
      }

      // NodeCSR tables (vsplit | ind ptrs | lst ptrs), one copy per GPU.
      {
        const size_t b_vs  = (ngpus_ + 1) * sizeof(GraphElem);
        const size_t b_ptr = ngpus_ * sizeof(const GraphElem*);
        std::vector<char> tab(b_vs + 2 * b_ptr);
        std::memcpy(tab.data(), h_vsplit.data(), b_vs);
        std::memcpy(tab.data() + b_vs, h_ind.data(), b_ptr);
        std::memcpy(tab.data() + b_vs + b_ptr, h_lst.data(), b_ptr);
        ncsr_.assign(ngpus_, NodeCSR{});
        for (int h = 0; h < ngpus_; h++) {
          Gpu& G = gpus_[h];
          CudaCall(cudaSetDevice(G.dev));
          CudaCall(cudaMalloc(&G.d_tab, tab.size()));
          CudaCall(cudaMemcpy(G.d_tab, tab.data(), tab.size(), cudaMemcpyHostToDevice));
          char* d = static_cast<char*>(G.d_tab);
          ncsr_[h].ngpus  = ngpus_;
          ncsr_[h].vsplit = reinterpret_cast<const GraphElem*>(d);
          ncsr_[h].ind    = reinterpret_cast<const GraphElem* const*>(d + b_vs);
          ncsr_[h].lst    = reinterpret_cast<const GraphElem* const*>(d + b_vs + b_ptr);
        }
      }

      // 3. Per-GPU streams, events, counters, partition table.
      for (int h = 0; h < ngpus_; h++) {
        Gpu& G = gpus_[h];
        CudaCall(cudaSetDevice(G.dev));
        CudaCall(cudaStreamCreateWithFlags(&G.s_cmp, cudaStreamNonBlocking));
        CudaCall(cudaStreamCreateWithFlags(&G.s_gen, cudaStreamNonBlocking));
        CudaCall(cudaStreamCreateWithFlags(&G.s_rcv, cudaStreamNonBlocking));
        CudaCall(cudaEventCreateWithFlags(&G.ev_gen, cudaEventDisableTiming));
        CudaCall(cudaEventCreateWithFlags(&G.ev_rcv[0], cudaEventDisableTiming));
        CudaCall(cudaEventCreateWithFlags(&G.ev_rcv[1], cudaEventDisableTiming));
        CudaCall(cudaMalloc(&G.d_count, sizeof(VmmCount)));
        CudaCall(cudaMemset(G.d_count, 0, sizeof(VmmCount)));
        CudaCall(cudaMalloc(&G.d_parts, (size_ + 1) * sizeof(GraphElem)));
        CudaCall(cudaMemcpy(G.d_parts, g_->get_parts(), (size_ + 1) * sizeof(GraphElem),
                           cudaMemcpyHostToDevice));
      }

      if (size_ == 1) {  // everything is node-local
        report_setup(MPI_Wtime() - t0, 0, max_elems * sizeof(GraphElem), 0);
        return;
      }

      // 4. Optional erange (min/max neighbour of every vertex), replicated per
      //    node but striped over the node's GPUs in one VMM range. Built on the
      //    GPUs, reduced across nodes with CUDA-aware MPI (no host copy).
      const GraphElem* d_erange = nullptr;
#if !defined(XXTRIC_VMM_NO_ERANGE_FILTER)
      d_erange = build_erange();
#endif

      // 5. GPU-built remote plans: exact #pairs per (GPU -> node) stream.
      GraphElem pairs_out = 0;
      for (int h = 0; h < ngpus_; h++) {
        Gpu& G = gpus_[h];
        CudaCall(cudaSetDevice(G.dev));
        vmm_plan_remote(ncsr_[h], h, G.ne, G.d_parts, size_, rank_, d_erange, G.plan, G.s_gen);
        G.dbeg.assign(size_ + 1, 0);
        G.doff.assign(size_, 0);
        G.dtot.assign(size_, 0);
        CudaCall(cudaMemcpy(G.dbeg.data(), G.plan.dbeg, (size_ + 1) * sizeof(GraphElem), cudaMemcpyDeviceToHost));
        CudaCall(cudaMemcpy(G.doff.data(), G.plan.doff, size_ * sizeof(GraphElem), cudaMemcpyDeviceToHost));
        CudaCall(cudaMemcpy(G.dtot.data(), G.plan.dtot, size_ * sizeof(GraphElem), cudaMemcpyDeviceToHost));
        pairs_out += G.plan.npairs;
      }

      // 6. Stream totals: sendT[s][h] = pairs GPU h sends to node s.
      //    Stride = max #GPUs over all nodes (nodes may differ).
      std::vector<GraphElem> sendT((size_t)size_ * maxg_, 0), recvT((size_t)size_ * maxg_, 0);
      for (int h = 0; h < ngpus_; h++)
        for (int s = 0; s < size_; s++) sendT[(size_t)s * maxg_ + h] = gpus_[h].dtot[s];
      MPI_Alltoall(sendT.data(), maxg_, MPI_GRAPH_TYPE, recvT.data(), maxg_, MPI_GRAPH_TYPE, comm_);

      // Pairs per message (cap). Must be identical on all ranks: sender and
      // receiver derive message sizes from it independently.
      //
      // The comm chunk of GPU h holds one slot per non-empty stream (outgoing
      // and incoming), each nsets(<=2) x cap pairs x 2 GraphElem. Sizing slots
      // by stream totals (cap = "unlimited") makes the chunk as large as all
      // inter-node wedge pairs of the GPU -- typically many times the edge
      // count -- and every GPU then allocates the MAX over all ranks
      // (DistributedBuffer agrees on one chunk size). So cap is bounded by a
      // per-GPU memory budget:
      //     max_h nslots_h * 2 * cap * 2 * sizeof(GraphElem) <= min_h budget_h
      // budget_h = XXTRIC_VMM_COMM_MB (env) if set, else
      //            XXTRIC_VMM_COMM_FRAC (env, default 0.5) x free memory now.
      std::vector<int> nslots(ngpus_, 0);
      for (int h = 0; h < ngpus_; h++)
        for (int s = 0; s < size_; s++) nslots[h] += (gpus_[h].dtot[s] > 0);
      for (int s = 0; s < size_; s++)
        for (int gp = 0; gp < maxg_; gp++)
          nslots[gp % ngpus_] += (recvT[(size_t)s * maxg_ + gp] > 0);

      unsigned long long budget = comm_budget_bytes();
      int maxslots = *std::max_element(nslots.begin(), nslots.end());
      MPI_Allreduce(MPI_IN_PLACE, &budget, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm_);
      MPI_Allreduce(MPI_IN_PLACE, &maxslots, 1, MPI_INT, MPI_MAX, comm_);

      long long capll = (bufsize > 0) ? std::max<long long>(1, (long long)bufsize / 2)
                                      : std::numeric_limits<long long>::max();
      if (maxslots > 0) {
        const unsigned long long per_pair = 2ull * 2ull * sizeof(GraphElem);  // 2 sets x (v,w)
        const unsigned long long slack    = (unsigned long long)maxslots * kAlign;
        const long long mem_cap = budget > slack
            ? (long long)((budget - slack) / ((unsigned long long)maxslots * per_pair)) : 0;
        if (mem_cap < 1024) {
          if (rank_ == 0)
            std::cerr << "*Error*: comm budget " << budget / 1048576.0 << " MB/GPU is too small for "
                      << maxslots << " streams per GPU; raise XXTRIC_VMM_COMM_MB/_FRAC or use more GPUs."
                      << std::endl;
          MPI_Abort(comm_, -99);
        }
        capll = std::min(capll, mem_cap);
      }
      MPI_Allreduce(MPI_IN_PLACE, &capll, 1, MPI_LONG_LONG, MPI_MIN, comm_);
      cap_pairs_ = (GraphElem)std::min<long long>(capll, std::numeric_limits<GraphElem>::max());
      comm_budget_ = budget;
      max_slots_   = maxslots;

      // 7. Lay out each GPU's comm chunk: [out slots | in slots], each slot
      //    = nsets x cap pairs. Double-buffered only if the stream needs >1 round.
      std::vector<size_t> used(ngpus_, 0);
      auto make = [&](GraphElem tot, GraphElem& cap, int& rounds, int& nsets, size_t& off, int h) {
        cap    = std::min(cap_pairs_, tot);
        rounds = (int)((tot + cap - 1) / cap);
        nsets  = rounds > 1 ? 2 : 1;
        off    = used[h];
        used[h] = align_up(used[h] + (size_t)nsets * (size_t)cap * 2 * sizeof(GraphElem));
      };
      for (int h = 0; h < ngpus_; h++)
        for (int s = 0; s < size_; s++) {
          const GraphElem tot = gpus_[h].dtot[s];
          if (tot <= 0) continue;
          OutSlot o{h, s, tot, 0, 0, 0, 0};
          make(tot, o.cap, o.rounds, o.nsets, o.off, h);
          outs_.push_back(o);
        }
      for (int s = 0; s < size_; s++)
        for (int gp = 0; gp < maxg_; gp++) {
          const GraphElem tot = recvT[(size_t)s * maxg_ + gp];
          if (tot <= 0) continue;
          InSlot in{gp % ngpus_, s, gp, tot, 0, 0, 0, 0};  // rail-aligned: GPU gp -> GPU gp mod G
          make(tot, in.cap, in.rounds, in.nsets, in.off, in.g);
          ins_.push_back(in);
        }
      size_t comm_bytes = 1;
      for (int h = 0; h < ngpus_; h++) comm_bytes = std::max(comm_bytes, used[h]);

      // 8. Comm buffers: one RDMA-capable VMM chunk per GPU, all GPUs of all
      //    nodes (vmm::DistributedBuffer; collective, sizes agreed inside).
      {
        mem_report("comm buffer", comm_bytes);
        vmm::DistributedBufferOptions o;
        o.bytesPerDevice        = comm_bytes;
        o.devices               = devs_;
        o.requireFullPeerAccess = true;
        o.gpuDirectRDMA         = kGDR;
        o.requireOneRankPerNode = one_per_node;
        comm_buf_ = std::make_unique<vmm::DistributedBuffer>(comm_, o);
      }

      // 9. Device-side send-slot tables (read by the generation kernel).
      max_rounds_ = 0;
      for (const auto& o : outs_) max_rounds_ = std::max(max_rounds_, o.rounds);
      for (const auto& i : ins_)  max_rounds_ = std::max(max_rounds_, i.rounds);
      for (int h = 0; h < ngpus_; h++) {
        Gpu& G = gpus_[h];
        std::vector<VmmSendSlot> slots;
        for (const auto& o : outs_) {
          if (o.g != h) continue;
          VmmSendSlot S;
          S.dbeg = G.dbeg[o.dest];
          S.dend = G.dbeg[o.dest + 1];
          S.doff = G.doff[o.dest];
          S.tot = o.tot;
          S.cap = o.cap;
          S.bufoff = (GraphElem)(o.off / sizeof(GraphElem));
          S.nsets = o.nsets;
          S.dest = o.dest;
          slots.push_back(S);
          G.max_cap = std::max(G.max_cap, o.cap);
        }
        G.nslots = (int)slots.size();
        if (G.nslots) {
          CudaCall(cudaSetDevice(G.dev));
          CudaCall(cudaMalloc(&G.d_slots, slots.size() * sizeof(VmmSendSlot)));
          CudaCall(cudaMemcpy(G.d_slots, slots.data(), slots.size() * sizeof(VmmSendSlot),
                             cudaMemcpyHostToDevice));
        }
      }
      MPI_Allreduce(&max_rounds_, &global_rounds_, 1, MPI_INT, MPI_MAX, comm_);

      report_setup(MPI_Wtime() - t0, pairs_out, max_elems * sizeof(GraphElem),
                   comm_buf_->chunkSize());
    }

    const GraphElem* build_erange()
    {
      const GraphElem nv = g_->get_nv();
      const size_t total = (size_t)nv * 2;
      const size_t per   = (total + ngpus_ - 1) / ngpus_;

      mem_report("erange (replicated min/max neighbours)", std::max<size_t>(per, 1) * sizeof(GraphElem));
      vmm::MultiGpuBufferOptions o;
      o.bytesPerDevice        = std::max<size_t>(per, 1) * sizeof(GraphElem);
      o.devices               = devs_;
      o.requireFullPeerAccess = true;
      o.gpuDirectRDMA         = kGDR;  // reduced in place by CUDA-aware MPI
      erange_.emplace(o);

      for (int h = 0; h < ngpus_; h++) {
        CudaCall(cudaSetDevice(devs_[h]));
        CudaCall(cudaMemsetAsync(erange_->chunk(h), 0, erange_->chunkSize(), gpus_[h].s_cmp));
      }
      sync_all(&Gpu::s_cmp);  // all chunks zeroed before any GPU writes into them
      GraphElem* er = erange_->as<GraphElem>();
      for (int h = 0; h < ngpus_; h++) {
        CudaCall(cudaSetDevice(devs_[h]));
        vmm_build_erange(ncsr_[h], h, gpus_[h].nv, er, gpus_[h].s_cmp);
      }
      sync_all(&Gpu::s_cmp);  // MPI is not stream-ordered

      // In-place SUM per physical chunk (one allocation per MPI call).
      const size_t ce = erange_->chunkSize() / sizeof(GraphElem);
      const size_t piece = size_t(1) << 28;
      for (int h = 0; h < ngpus_; h++) {
        const size_t b = (size_t)h * ce, e = std::min(total, b + ce);
        for (size_t x = b; x < e; x += piece) {
          const int n = (int)std::min(piece, e - x);
          MPI_Allreduce(MPI_IN_PLACE, er + x, n, MPI_GRAPH_TYPE, MPI_SUM, comm_);
        }
      }
      return er;
    }

    /// Smallest free device memory on this node right now (bytes).
    size_t min_free_bytes()
    {
      size_t mn = std::numeric_limits<size_t>::max();
      for (int d : devs_) {
        size_t fr = 0, tot = 0;
        CudaCall(cudaSetDevice(d));
        CudaCall(cudaMemGetInfo(&fr, &tot));
        mn = std::min(mn, fr);
      }
      return mn;
    }

    /// Per-GPU byte budget for the comm chunk (see setup step 6).
    unsigned long long comm_budget_bytes()
    {
      if (const char* mb = std::getenv("XXTRIC_VMM_COMM_MB"))
        return (unsigned long long)std::max(1.0, std::atof(mb)) * 1048576ull;
      double frac = 0.5;
      if (const char* f = std::getenv("XXTRIC_VMM_COMM_FRAC")) frac = std::atof(f);
      frac = std::min(0.95, std::max(0.01, frac));
      return (unsigned long long)(frac * (double)min_free_bytes());
    }

    /// Collective. Prints (rank 0) the per-GPU size about to be allocated,
    /// max over ranks, against the smallest free memory over all GPUs.
    void mem_report(const char* what, size_t bytes)
    {
      unsigned long long b = bytes, bmax = 0;
      unsigned long long fr = min_free_bytes(), frmin = 0;
      MPI_Reduce(&b, &bmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm_);
      MPI_Reduce(&fr, &frmin, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, 0, comm_);
      if (rank_ == 0)
        std::cout << "VMM alloc [" << what << "]: " << bmax / 1048576.0
                  << " MB per GPU (max over ranks); min free GPU memory "
                  << frmin / 1048576.0 << " MB" << std::endl;
    }

    void sync_all(cudaStream_t Gpu::*s)
    {
      for (auto& G : gpus_) {
        CudaCall(cudaSetDevice(G.dev));
        CudaCall(cudaStreamSynchronize(G.*s));
      }
    }

    void report_setup(double t, GraphElem pairs_out, size_t csr_bytes, size_t comm_bytes)
    {
      double tmax = 0.0;
      unsigned long long p = (unsigned long long)pairs_out, psum = 0, pmax = 0;
      MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
      MPI_Reduce(&p, &psum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, comm_);
      MPI_Reduce(&p, &pmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, comm_);
      if (rank_ == 0) {
        std::cout << "VMM setup: " << size_ << " node(s) x " << ngpus_ << " GPU(s), max time (secs.): "
                  << tmax << std::endl;
        std::cout << "VMM CSR chunk/GPU (MB): " << csr_bytes / 1048576.0
                  << ", comm chunk/GPU (MB): " << comm_bytes / 1048576.0 << std::endl;
        std::cout << "Inter-node wedge pairs: total " << psum << ", max per node " << pmax
                  << ", comm budget/GPU (MB): " << comm_budget_ / 1048576.0
                  << ", max streams/GPU: " << max_slots_
                  << ", pairs per message: "
                  << (cap_pairs_ == std::numeric_limits<GraphElem>::max() ? std::string("unlimited")
                                                                           : std::to_string(cap_pairs_))
                  << std::endl;
      }
    }

    // --------------------------------------------------------------- exchange
    //
    // Round k of every stream uses set b = k % nsets of its slot. Overlap:
    //   recv kernels of round k  || exchange of round k+1
    //   sends of round k         || generation/exchange of round k+1
    // Reuse of a set at round k+2 waits for the round-k sends (MPI) or the
    // round-k receive kernels (CUDA events).

    void post_recvs(int k, std::vector<MPI_Request>& rreq, std::vector<int>& rslot,
                    std::vector<int>& rleft)
    {
      for (size_t i = 0; i < ins_.size(); i++) {
        const InSlot& in = ins_[i];
        const GraphElem len = round_len(in.tot, in.cap, k);
        if (len <= 0) continue;
        const size_t off = in.off + (size_t)(k % in.nsets) * in.cap * 2 * sizeof(GraphElem);
        const size_t before = rreq.size();
        comm_buf_->irecv(chunk_of(in.g), off, (size_t)len * 2 * sizeof(GraphElem), in.src,
                         tag_of(in.srcgpu), rreq);
        rslot.insert(rslot.end(), rreq.size() - before, (int)i);
        rleft[i] = (int)(rreq.size() - before);
      }
    }

    void post_sends(int k, int h, std::vector<MPI_Request>& sreq)
    {
      for (const OutSlot& o : outs_) {
        if (o.g != h) continue;
        const GraphElem len = round_len(o.tot, o.cap, k);
        if (len <= 0) continue;
        const size_t off = o.off + (size_t)(k % o.nsets) * o.cap * 2 * sizeof(GraphElem);
        comm_buf_->isend(chunk_of(h), off, (size_t)len * 2 * sizeof(GraphElem), o.dest,
                         tag_of(h), sreq);
      }
    }

    void exchange_and_count()
    {
      std::vector<MPI_Request> sreq[2];
      std::vector<MPI_Request> rreq;
      std::vector<int> rslot, rleft(ins_.size(), 0), idx;
      std::vector<char> gen_active(ngpus_, 0);

      for (int k = 0; k < max_rounds_; k++) {
        const int b = k & 1;

        // (a) receive sets of round k-2 consumed?
        if (k >= 2)
          for (auto& G : gpus_) {
            CudaCall(cudaSetDevice(G.dev));
            CudaCall(cudaEventSynchronize(G.ev_rcv[b]));
          }

        // (b) receives first: data lands directly in GPU memory
        rreq.clear(); rslot.clear();
        std::fill(rleft.begin(), rleft.end(), 0);
        post_recvs(k, rreq, rslot, rleft);

        // (c) send sets of round k-2 drained?
        if (!sreq[b].empty()) {
          MPI_Waitall((int)sreq[b].size(), sreq[b].data(), MPI_STATUSES_IGNORE);
          sreq[b].clear();
        }

        // (d) every GPU fills its round-k send buffers (no host buffering)
        for (int h = 0; h < ngpus_; h++) {
          Gpu& G = gpus_[h];
          gen_active[h] = 0;
          if (!G.nslots) continue;
          bool any = false;
          for (const OutSlot& o : outs_)
            if (o.g == h && round_len(o.tot, o.cap, k) > 0) { any = true; break; }
          if (!any) continue;
          CudaCall(cudaSetDevice(G.dev));
          GraphElem* sbuf = static_cast<GraphElem*>(comm_buf_->localPtr(chunk_of(h)));
          vmm_generate_pairs(ncsr_[h], h, G.plan, G.d_slots, G.nslots, G.max_cap, k, sbuf, G.s_gen);
          CudaCall(cudaEventRecord(G.ev_gen, G.s_gen));
          gen_active[h] = 1;
        }

        // (e) post sends as each GPU's buffers complete
        //     [single host post point -> replace with a device-triggered transport]
        for (int h = 0; h < ngpus_; h++) {
          if (!gen_active[h]) continue;
          CudaCall(cudaSetDevice(gpus_[h].dev));
          CudaCall(cudaEventSynchronize(gpus_[h].ev_gen));
          post_sends(k, h, sreq[b]);
        }

        // (f) count each incoming stream on its GPU as soon as it has landed
        int pending = 0;
        for (int c : rleft) pending += (c > 0);
        idx.resize(rreq.size());
        while (pending > 0) {
          int nout = 0;
          MPI_Waitsome((int)rreq.size(), rreq.data(), &nout, idx.data(), MPI_STATUSES_IGNORE);
          if (nout == MPI_UNDEFINED) break;
          for (int t = 0; t < nout; t++) {
            const int i = rslot[idx[t]];
            if (--rleft[i] > 0) continue;
            const InSlot& in = ins_[i];
            Gpu& G = gpus_[in.g];
            const size_t off = in.off + (size_t)(k % in.nsets) * in.cap * 2 * sizeof(GraphElem);
            CudaCall(cudaSetDevice(G.dev));
            vmm_count_recv(ncsr_[in.g], static_cast<const GraphElem*>(comm_buf_->localPtr(chunk_of(in.g), off)),
                           round_len(in.tot, in.cap, k), G.d_count, G.s_rcv);
            --pending;
          }
        }
        for (auto& G : gpus_) {
          CudaCall(cudaSetDevice(G.dev));
          CudaCall(cudaEventRecord(G.ev_rcv[b], G.s_rcv));
        }
      }

      for (auto& s : sreq)
        if (!s.empty()) MPI_Waitall((int)s.size(), s.data(), MPI_STATUSES_IGNORE);
    }

    // ------------------------------------------------------------------ state
    Graph* g_;
    MPI_Comm comm_;
    int rank_ = 0, size_ = 1;
    bool cleared_ = false;

    std::vector<int> devs_;
    int ngpus_ = 0, maxg_ = 0;
    std::vector<Gpu> gpus_;
    std::vector<NodeCSR> ncsr_;  // per GPU: tables in that GPU's memory

    std::optional<vmm::MultiGpuBuffer> csr_;      // node CSR, striped over GPUs
    std::optional<vmm::MultiGpuBuffer> erange_;   // node copy of min/max neighbours
    std::unique_ptr<vmm::DistributedBuffer> comm_buf_;  // send/recv slots, all GPUs, all nodes

    std::vector<OutSlot> outs_;
    std::vector<InSlot> ins_;
    GraphElem cap_pairs_ = std::numeric_limits<GraphElem>::max();
    int max_rounds_ = 0, global_rounds_ = 0, max_slots_ = 0;
    unsigned long long comm_budget_ = 0;
    GraphElem ntriangles_ = 0;
};

#endif
