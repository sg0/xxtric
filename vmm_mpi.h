// DistributedBuffer: one logical array striped across ALL GPUs of ALL nodes.
//
//   global chunk:  0      1      2      3    |   4      5      6      7    | ...
//   owner:         node 0 (rank 0)           |   node 1 (rank 1)           |
//                  GPU0   GPU1   GPU2   GPU3 |   GPU0   GPU1   GPU2   GPU3 |
//                  \______ VMM range ______/ |   \______ VMM range ______/
//
// * Within a node, the node's chunks live in one vmm::MultiGpuBuffer: a single
//   contiguous VMM virtual range mapped for every GPU of the node, so any
//   local GPU can load/store any local chunk directly (zero copy, NVLink/PCIe
//   P2P), and local chunk-to-chunk transfers are P2P DMA over those mappings.
// * Between nodes, data moves with nonblocking MPI point-to-point
//   (MPI_Isend / MPI_Irecv) issued directly on the VMM device pointers.
//   This requires a CUDA-aware (GPU-aware) MPI; no host staging buffers.
//
// Process model:
//   ONE MPI rank per node. That rank owns and drives all of the node's GPUs
//   (or the subset given in options.devices). Chunks are numbered globally
//   in rank order: rank r owns chunks [firstChunkOf(r), firstChunkOf(r+1)).
//   Nodes may have different GPU counts; the chunk size is the same for all.
//
// Usage:
//   MPI_Init(...);
//   {
//     vmm::DistributedBufferOptions opt;
//     opt.bytesPerDevice = 64 << 20;
//     vmm::DistributedBuffer buf(MPI_COMM_WORLD, opt);   // collective
//
//     float* mine = buf.localAs<float>(buf.firstLocalChunk());  // zero copy
//
//     std::vector<vmm::Transfer> plan = {...};   // same list on every rank
//     vmm::Exchange ex = buf.exchange(plan);     // routes P2P vs MPI
//     ex.wait();
//   }                                            
//   MPI_Finalize();
//
// Rules:
//   * CUDA-aware MPI that accepts CUDA VMM (cuMemCreate/cuMemMap) memory.
//     For GPUDirect RDMA the chunks are allocated with
//     allocFlags.gpuDirectRDMACapable (option gpuDirectRDMA, default on).
//   * MPI calls are not stream-ordered: kernels that produce data for an
//     exchange() must complete (e.g. cudaDeviceSynchronize)
//     before exchange() invoked, and consumers must run after wait().
//   * Not thread-safe; use from one thread per rank (MPI_THREAD_SINGLE ok).

#pragma once

#include "vmm_multigpu.h"

#include <cuda.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vmm {

// ===========================================================================
// Public types
// ===========================================================================

struct DistributedBufferOptions {
    // Bytes of physical memory per GPU. Every rank should pass the same value;
    // the maximum over all ranks is used, rounded up to an allocation
    // granularity that is valid on every node.
    std::size_t bytesPerDevice = 0;

    // This node's CUDA device ordinals, in chunk order. Empty = all visible.
    std::vector<int> devices;

    // Passed through to the node-local MultiGpuBuffer.
    bool requireFullPeerAccess = true;
    bool useRecommendedGranularity = true;

    // Allocate GPUDirect RDMA-capable memory so the NIC can access the chunks
    // directly. Construction fails if a GPU does not support it.
    bool gpuDirectRDMA = true;

    // Fail if more than one rank of the communicator runs on the same node.
    // Set to false only for testing (e.g. several ranks on one machine with
    // CUDA_VISIBLE_DEVICES split between them); then "remote" chunks on the
    // same node are reached via MPI instead of VMM.
    bool requireOneRankPerNode = true;
};

// One copy of `bytes` from (srcChunk, srcOffset) to (dstChunk, dstOffset),
// chunk indices are global. The region must lie within a single chunk.
struct Transfer {
    int srcChunk = 0;
    std::size_t srcOffset = 0;
    int dstChunk = 0;
    std::size_t dstOffset = 0;
    std::size_t bytes = 0;
    int tag = 0;  // MPI tag used if the transfer crosses nodes
};

// Handle for a batch of in-flight transfers returned by exchange().
// Must not outlive the DistributedBuffer that created it. Call wait();
// if it is destroyed while still pending, the destructor waits.
class Exchange {
public:
    Exchange() = default;
    ~Exchange() {
        if (pending()) {
            try { wait(); } catch (...) {}
        }
    }

    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;
    Exchange(Exchange&& o) noexcept { swap(o); }
    Exchange& operator=(Exchange&& o) noexcept {
        if (this != &o) {
            if (pending()) {
                try { wait(); } catch (...) {}
            }
            swap(o);
        }
        return *this;
    }

    // Completes all MPI requests and all intra-node copies of this batch.
    void wait();
    bool pending() const noexcept { return !requests_.empty() || !streams_.empty(); }

    // Transfers of this batch handled by this rank, by path.
    int localCopies() const noexcept { return localCopies_; }  // VMM P2P, same node
    int sends() const noexcept { return sends_; }              // MPI_Isend
    int recvs() const noexcept { return recvs_; }              // MPI_Irecv

private:
    friend class DistributedBuffer;
    struct StreamRef {
        CUcontext ctx;
        CUstream stream;
    };
    void swap(Exchange& o) noexcept {
        std::swap(requests_, o.requests_);
        std::swap(streams_, o.streams_);
        std::swap(localCopies_, o.localCopies_);
        std::swap(sends_, o.sends_);
        std::swap(recvs_, o.recvs_);
    }

    std::vector<MPI_Request> requests_;
    std::vector<StreamRef> streams_;
    int localCopies_ = 0;
    int sends_ = 0;
    int recvs_ = 0;
};

class DistributedBuffer {
public:
    // Collective over `comm`.
    DistributedBuffer(MPI_Comm comm, const DistributedBufferOptions& options);
    ~DistributedBuffer() { release(); }

    DistributedBuffer(const DistributedBuffer&) = delete;
    DistributedBuffer& operator=(const DistributedBuffer&) = delete;
    DistributedBuffer(DistributedBuffer&&) = delete;  // hold in a unique_ptr
    DistributedBuffer& operator=(DistributedBuffer&&) = delete;  // if needed

    // ---- MPI ---------------------------------------------------------------
    MPI_Comm comm() const noexcept { return comm_; }  // internal duplicate
    int rank() const noexcept { return rank_; }
    int size() const noexcept { return size_; }

    // ---- Global layout -----------------------------------------------------
    int globalChunks() const noexcept { return static_cast<int>(chunkDevice_.size()); }
    std::size_t chunkSize() const noexcept { return chunk_; }
    std::size_t globalSize() const noexcept { return chunk_ * chunkDevice_.size(); }

    int firstChunkOf(int rank) const { checkRank(rank); return firstChunk_[rank]; }
    int numChunksOf(int rank) const {
        checkRank(rank);
        return firstChunk_[rank + 1] - firstChunk_[rank];
    }
    int firstLocalChunk() const noexcept { return firstChunk_[rank_]; }
    int numLocalChunks() const noexcept { return firstChunk_[rank_ + 1] - firstChunk_[rank_]; }

    int ownerRank(int chunk) const;  // rank (node) that holds `chunk`
    int deviceOf(int chunk) const;   // CUDA ordinal of `chunk` on its node
    bool isLocal(int chunk) const { return ownerRank(chunk) == rank_; }

    // ---- Node-local, zero-copy access --------------------------------------
    // Pointer into a chunk owned by this rank. Valid on every GPU of this node
    // (direct loads/stores over NVLink/PCIe). Throws for remote chunks.
    void* localPtr(int chunk, std::size_t byteOffset = 0) const;
    template <typename T>
    T* localAs(int chunk) const { return static_cast<T*>(localPtr(chunk)); }

    // The node's VMM buffer: all local chunks back to back, in global order
    // starting at firstLocalChunk(). Gives one pointer over the whole node.
    MultiGpuBuffer& node() { return *node_; }
    const MultiGpuBuffer& node() const { return *node_; }

    // ---- Communication -----------------------------------------------------
    // SPMD batch transfer: EVERY rank calls this with the SAME list. Each
    // transfer is routed by locality:
    //   src and dst on this node  -> async P2P copy over the VMM mappings,
    //                                run by the destination GPU (no MPI)
    //   only dst on this node     -> MPI_Irecv straight into the VMM chunk
    //   only src on this node     -> MPI_Isend straight from the VMM chunk
    //   neither                   -> nothing
    // Returns immediately; call wait() on the result.
    // Source data must be ready (see rules above). Regions of the batch's
    // local copies must not overlap.
    Exchange exchange(const std::vector<Transfer>& transfers);

    // Low-level inter-node primitives on local chunks (GPU-aware MPI).
    // Messages larger than kMaxMessageBytes are split into several messages
    // with the same tag; the matching irecv/isend splits identically.
    void isend(int srcChunk, std::size_t srcOffset, std::size_t bytes, int dstRank, int tag,
               std::vector<MPI_Request>& requests) const;
    void irecv(int dstChunk, std::size_t dstOffset, std::size_t bytes, int srcRank, int tag,
               std::vector<MPI_Request>& requests) const;

    static constexpr std::size_t kMaxMessageBytes = std::size_t(1) << 30;  // 1 GiB

private:
    void release() noexcept;
    void agree(bool localOk, const std::string& localError, const char* stage);
    void checkChunk(int chunk) const;
    void checkRank(int rank) const;
    void checkRegion(int chunk, std::size_t offset, std::size_t bytes, const char* who) const;
    void checkTag(int tag) const;

    MPI_Comm comm_ = MPI_COMM_NULL;
    int rank_ = 0;
    int size_ = 0;
    int tagUB_ = 32767;  // MPI guarantees at least this

    std::size_t chunk_ = 0;
    std::vector<int> firstChunk_;   // size_ + 1 prefix sums of chunks per rank
    std::vector<int> chunkDevice_;  // per global chunk: CUDA ordinal on its node

    std::optional<MultiGpuBuffer> node_;
    std::vector<CUdevice> cuDevices_;  // per local chunk
    std::vector<CUcontext> contexts_;  // retained primary contexts
    std::vector<CUstream> streams_;    // copy stream per local GPU
};

// ===========================================================================
// Implementation (inline, header-only)
// ===========================================================================
namespace mpi_detail {

inline void check(int rc, const char* what) {
    if (rc == MPI_SUCCESS) return;
    char msg[MPI_MAX_ERROR_STRING];
    int len = 0;
    if (MPI_Error_string(rc, msg, &len) != MPI_SUCCESS) len = 0;
    throw std::runtime_error(std::string(what) + " failed: " + std::string(msg, len));
}

inline void cuCheck(CUresult r, const char* what) {
    ::vmm::detail::check(r, what, __FILE__, __LINE__);
}

}  // namespace mpi_detail

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
inline DistributedBuffer::DistributedBuffer(MPI_Comm comm, const DistributedBufferOptions& opt) {
    int initialized = 0;
    mpi_detail::check(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized) throw std::logic_error("DistributedBuffer: MPI is not initialized");

    mpi_detail::check(MPI_Comm_dup(comm, &comm_), "MPI_Comm_dup");
    try {
        mpi_detail::check(MPI_Comm_rank(comm_, &rank_), "MPI_Comm_rank");
        mpi_detail::check(MPI_Comm_size(comm_, &size_), "MPI_Comm_size");
        {
            void* v = nullptr;
            int flag = 0;
            mpi_detail::check(MPI_Comm_get_attr(comm_, MPI_TAG_UB, &v, &flag),
                              "MPI_Comm_get_attr(MPI_TAG_UB)");
            if (flag && v) tagUB_ = *static_cast<int*>(v);
        }

        // 1. One rank per node?
        {
            MPI_Comm nodeComm;
            mpi_detail::check(MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, rank_,
                                                  MPI_INFO_NULL, &nodeComm),
                              "MPI_Comm_split_type");
            int ranksOnNode = 0;
            mpi_detail::check(MPI_Comm_size(nodeComm, &ranksOnNode), "MPI_Comm_size");
            mpi_detail::check(MPI_Comm_free(&nodeComm), "MPI_Comm_free");
            const bool ok = !(opt.requireOneRankPerNode && ranksOnNode > 1);
            agree(ok,
                  std::to_string(ranksOnNode) +
                      " ranks share this node; launch one rank per node "
                      "(e.g. mpirun --map-by ppr:1:node, srun --ntasks-per-node=1) "
                      "or set requireOneRankPerNode = false for testing",
                  "topology");
        }

        // 2. Local devices and their allocation granularity.
        std::vector<int> devices;
        unsigned long long granularity = 1;
        {
            bool ok = true;
            std::string err;
            try {
                if (opt.bytesPerDevice == 0)
                    throw std::invalid_argument("bytesPerDevice must be > 0");
                devices = MultiGpuBuffer::resolveDevices(opt.devices);
                for (int d : devices)
                    granularity = std::lcm(
                        granularity,
                        static_cast<unsigned long long>(MultiGpuBuffer::allocationGranularity(
                            d, opt.useRecommendedGranularity, opt.gpuDirectRDMA)));
            } catch (const std::exception& e) {
                ok = false;
                err = e.what();
            }
            agree(ok, err, "device discovery");
        }

        // 3. Global layout: chunks per rank, device ordinals, common chunk size.
        {
            const int nLocal = static_cast<int>(devices.size());
            std::vector<int> counts(size_);
            mpi_detail::check(MPI_Allgather(&nLocal, 1, MPI_INT, counts.data(), 1, MPI_INT, comm_),
                              "MPI_Allgather");
            firstChunk_.assign(size_ + 1, 0);
            for (int r = 0; r < size_; ++r) firstChunk_[r + 1] = firstChunk_[r] + counts[r];

            chunkDevice_.resize(firstChunk_[size_]);
            mpi_detail::check(MPI_Allgatherv(devices.data(), nLocal, MPI_INT, chunkDevice_.data(),
                                             counts.data(), firstChunk_.data(), MPI_INT, comm_),
                              "MPI_Allgatherv");

            std::vector<unsigned long long> grans(size_);
            mpi_detail::check(MPI_Allgather(&granularity, 1, MPI_UNSIGNED_LONG_LONG, grans.data(),
                                            1, MPI_UNSIGNED_LONG_LONG, comm_),
                              "MPI_Allgather");
            unsigned long long common = 1;
            for (unsigned long long g : grans) common = std::lcm(common, g);

            unsigned long long requested = opt.bytesPerDevice, maxRequested = 0;
            mpi_detail::check(MPI_Allreduce(&requested, &maxRequested, 1, MPI_UNSIGNED_LONG_LONG,
                                            MPI_MAX, comm_),
                              "MPI_Allreduce");
            chunk_ = static_cast<std::size_t>(((maxRequested + common - 1) / common) * common);
        }

        // 4. Allocate + map this node's chunks (VMM), set up copy streams.
        {
            bool ok = true;
            std::string err;
            try {
                MultiGpuBufferOptions nopt;
                nopt.bytesPerDevice            = chunk_;
                nopt.devices                   = devices;
                nopt.requireFullPeerAccess     = opt.requireFullPeerAccess;
                nopt.useRecommendedGranularity = opt.useRecommendedGranularity;
                nopt.gpuDirectRDMA             = opt.gpuDirectRDMA;
                node_.emplace(nopt);
                if (node_->chunkSize() != chunk_)
                    throw std::logic_error("node chunk size differs from the global chunk size");

                for (int d : devices) {
                    CUdevice dev;
                    mpi_detail::cuCheck(cuDeviceGet(&dev, d), "cuDeviceGet");
                    CUcontext ctx;
                    mpi_detail::cuCheck(cuDevicePrimaryCtxRetain(&ctx, dev),
                                        "cuDevicePrimaryCtxRetain");
                    cuDevices_.push_back(dev);
                    contexts_.push_back(ctx);
                    detail::ScopedContext guard(ctx);
                    CUstream s;
                    mpi_detail::cuCheck(cuStreamCreate(&s, CU_STREAM_NON_BLOCKING),
                                        "cuStreamCreate");
                    streams_.push_back(s);
                }
            } catch (const std::exception& e) {
                ok = false;
                err = e.what();
            }
            agree(ok, err, "allocation");
        }
    } catch (...) {
        release();
        throw;
    }
}

// Every rank reaches agree() at the same point, so a local failure turns into
// an exception on all ranks instead of a hang in the next collective.
inline void DistributedBuffer::agree(bool localOk, const std::string& localError,
                                     const char* stage) {
    int mine = localOk ? 1 : 0, all = 0;
    mpi_detail::check(MPI_Allreduce(&mine, &all, 1, MPI_INT, MPI_LAND, comm_), "MPI_Allreduce");
    if (all) return;

    int failing = localOk ? size_ : rank_, firstFailing = 0;
    mpi_detail::check(
        MPI_Allreduce(&failing, &firstFailing, 1, MPI_INT, MPI_MIN, comm_), "MPI_Allreduce");
    std::string msg = std::string("DistributedBuffer (") + stage + "): ";
    msg += localOk ? "failed on rank " + std::to_string(firstFailing)
                   : "rank " + std::to_string(rank_) + ": " + localError;
    throw std::runtime_error(msg);
}

inline void DistributedBuffer::release() noexcept {
    for (std::size_t i = 0; i < streams_.size(); ++i) {
        if (cuCtxPushCurrent(contexts_[i]) == CUDA_SUCCESS) {
            cuStreamDestroy(streams_[i]);
            CUcontext tmp;
            cuCtxPopCurrent(&tmp);
        }
    }
    streams_.clear();
    for (CUdevice d : cuDevices_) cuDevicePrimaryCtxRelease(d);
    cuDevices_.clear();
    contexts_.clear();

    node_.reset();  // unmap + free this node's VMM chunks

    if (comm_ != MPI_COMM_NULL) {
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Comm_free(&comm_);
        comm_ = MPI_COMM_NULL;
    }
}

// ---------------------------------------------------------------------------
// Layout queries
// ---------------------------------------------------------------------------
inline void DistributedBuffer::checkChunk(int chunk) const {
    if (chunk < 0 || chunk >= globalChunks())
        throw std::out_of_range("DistributedBuffer: chunk " + std::to_string(chunk) +
                                " out of range [0, " + std::to_string(globalChunks()) + ")");
}

inline void DistributedBuffer::checkRank(int rank) const {
    if (rank < 0 || rank >= size_)
        throw std::out_of_range("DistributedBuffer: rank " + std::to_string(rank) +
                                " out of range");
}

inline void DistributedBuffer::checkRegion(int chunk, std::size_t offset, std::size_t bytes,
                                           const char* who) const {
    checkChunk(chunk);
    if (offset > chunk_ || bytes > chunk_ - offset)
        throw std::out_of_range(std::string("DistributedBuffer::") + who + ": region [" +
                                std::to_string(offset) + ", +" + std::to_string(bytes) +
                                ") exceeds chunk size " + std::to_string(chunk_));
}

inline void DistributedBuffer::checkTag(int tag) const {
    if (tag < 0 || tag > tagUB_)
        throw std::out_of_range("DistributedBuffer: tag " + std::to_string(tag) +
                                " outside [0, MPI_TAG_UB=" + std::to_string(tagUB_) + "]");
}

inline int DistributedBuffer::ownerRank(int chunk) const {
    checkChunk(chunk);
    // Last rank whose first chunk is <= chunk.
    auto it = std::upper_bound(firstChunk_.begin(), firstChunk_.end(), chunk);
    return static_cast<int>(it - firstChunk_.begin()) - 1;
}

inline int DistributedBuffer::deviceOf(int chunk) const {
    checkChunk(chunk);
    return chunkDevice_[chunk];
}

inline void* DistributedBuffer::localPtr(int chunk, std::size_t byteOffset) const {
    checkRegion(chunk, byteOffset, 0, "localPtr");
    if (!isLocal(chunk))
        throw std::invalid_argument("DistributedBuffer::localPtr: chunk " +
                                    std::to_string(chunk) + " is owned by rank " +
                                    std::to_string(ownerRank(chunk)));
    return static_cast<char*>(node_->chunk(chunk - firstLocalChunk())) + byteOffset;
}

// ---------------------------------------------------------------------------
// Communication
// ---------------------------------------------------------------------------
inline void DistributedBuffer::isend(int srcChunk, std::size_t srcOffset, std::size_t bytes,
                                     int dstRank, int tag,
                                     std::vector<MPI_Request>& requests) const {
    checkRegion(srcChunk, srcOffset, bytes, "isend");
    checkRank(dstRank);
    checkTag(tag);
    const char* p = static_cast<const char*>(localPtr(srcChunk, srcOffset));
    std::size_t done = 0;
    do {  // a zero-byte transfer still sends one (empty) message
        const std::size_t n = std::min(kMaxMessageBytes, bytes - done);
        MPI_Request r;
        mpi_detail::check(MPI_Isend(p + done, static_cast<int>(n), MPI_BYTE, dstRank, tag, comm_,
                                    &r),
                          "MPI_Isend");
        requests.push_back(r);
        done += n;
    } while (done < bytes);
}

inline void DistributedBuffer::irecv(int dstChunk, std::size_t dstOffset, std::size_t bytes,
                                     int srcRank, int tag,
                                     std::vector<MPI_Request>& requests) const {
    checkRegion(dstChunk, dstOffset, bytes, "irecv");
    checkRank(srcRank);
    checkTag(tag);
    char* p = static_cast<char*>(localPtr(dstChunk, dstOffset));
    std::size_t done = 0;
    do {
        const std::size_t n = std::min(kMaxMessageBytes, bytes - done);
        MPI_Request r;
        mpi_detail::check(MPI_Irecv(p + done, static_cast<int>(n), MPI_BYTE, srcRank, tag, comm_,
                                    &r),
                          "MPI_Irecv");
        requests.push_back(r);
        done += n;
    } while (done < bytes);
}

inline Exchange DistributedBuffer::exchange(const std::vector<Transfer>& transfers) {
    // Validate the whole batch before posting anything.
    for (const Transfer& t : transfers) {
        checkRegion(t.srcChunk, t.srcOffset, t.bytes, "exchange(src)");
        checkRegion(t.dstChunk, t.dstOffset, t.bytes, "exchange(dst)");
        checkTag(t.tag);
    }

    Exchange ex;

    // 1. Inter-node receives first, so incoming data can land immediately.
    for (const Transfer& t : transfers) {
        if (isLocal(t.dstChunk) && !isLocal(t.srcChunk)) {
            irecv(t.dstChunk, t.dstOffset, t.bytes, ownerRank(t.srcChunk), t.tag, ex.requests_);
            ++ex.recvs_;
        }
    }

    // 2. Intra-node transfers: P2P DMA over the VMM mappings, pulled by the
    //    destination GPU on its copy stream. No MPI, no host buffers.
    std::vector<char> streamUsed(streams_.size(), 0);
    for (const Transfer& t : transfers) {
        if (isLocal(t.dstChunk) && isLocal(t.srcChunk)) {
            const int i = t.dstChunk - firstLocalChunk();
            detail::ScopedContext guard(contexts_[i]);
            mpi_detail::cuCheck(
                cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(localPtr(t.dstChunk, t.dstOffset)),
                                  reinterpret_cast<CUdeviceptr>(localPtr(t.srcChunk, t.srcOffset)),
                                  t.bytes, streams_[i]),
                "cuMemcpyDtoDAsync");
            streamUsed[i] = 1;
            ++ex.localCopies_;
        }
    }
    for (std::size_t i = 0; i < streams_.size(); ++i)
        if (streamUsed[i]) ex.streams_.push_back({contexts_[i], streams_[i]});

    // 3. Inter-node sends, straight from GPU memory (CUDA-aware MPI).
    for (const Transfer& t : transfers) {
        if (isLocal(t.srcChunk) && !isLocal(t.dstChunk)) {
            isend(t.srcChunk, t.srcOffset, t.bytes, ownerRank(t.dstChunk), t.tag, ex.requests_);
            ++ex.sends_;
        }
    }
    return ex;
}

inline void Exchange::wait() {
    if (!requests_.empty()) {
        const int rc = MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                                   MPI_STATUSES_IGNORE);
        requests_.clear();
        mpi_detail::check(rc, "MPI_Waitall");
    }
    std::vector<StreamRef> streams;
    streams.swap(streams_);
    for (const StreamRef& s : streams) {
        detail::ScopedContext guard(s.ctx);
        mpi_detail::cuCheck(cuStreamSynchronize(s.stream), "cuStreamSynchronize");
    }
}

}  // namespace vmm
