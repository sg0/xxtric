// MultiGpuBuffer: one contiguous virtual address range whose physical backing
// is striped across several GPUs, mapped with the CUDA Virtual Memory
// Management (VMM) API so that every participating GPU can load/store any part
// of it directly over NVLink / PCIe P2P (zero copy).
//
//   [ chunk 0 | chunk 1 | ... | chunk N-1 ]   <- one pointer, data()
//     GPU d0    GPU d1          GPU dN-1      <- physical backing
//
// Standaline usage:
//   #include "vmm_multigpu.h"
//
//   vmm::MultiGpuBufferOptions opt;
//   opt.bytesPerDevice = 64 << 20;            // rounded up to granularity
//   vmm::MultiGpuBuffer buf(opt);             // all visible GPUs
//   float* p = buf.as<float>();               // valid on every GPU
//   float* mine = static_cast<float*>(buf.chunk(i));  // backed by devices()[i]

#pragma once

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vmm {

// ===========================================================================
// Public types
// ===========================================================================

// Exception carrying the underlying CUDA driver error code.
class CudaError : public std::runtime_error {
public:
    CudaError(CUresult code, const std::string& what)
        : std::runtime_error(what), code_(code) {}
    CUresult code() const noexcept { return code_; }

private:
    CUresult code_;
};

enum class Access {
    None,       // CU_MEM_ACCESS_FLAGS_PROT_NONE
    Read,       // CU_MEM_ACCESS_FLAGS_PROT_READ
    ReadWrite   // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
};

struct MultiGpuBufferOptions {
    // Bytes of physical memory to place on each GPU. Rounded up to the
    // largest allocation granularity among the participating GPUs.
    std::size_t bytesPerDevice = 0;

    // CUDA device ordinals to stripe across, in chunk order.
    // Empty = all visible devices (0 .. count-1).
    std::vector<int> devices;

    // true:  throw if any pair of participating GPUs lacks P2P access.
    // false: each chunk is made accessible only to its owner and to the
    //        GPUs that can reach it via P2P; others get no mapping access.
    bool requireFullPeerAccess = true;

    // true: CU_MEM_ALLOC_GRANULARITY_RECOMMENDED (better performance)
    // false: CU_MEM_ALLOC_GRANULARITY_MINIMUM   (less padding)
    bool useRecommendedGranularity = true;

    // Request GPUDirect RDMA-capable physical memory
    // (CUmemAllocationProp::allocFlags.gpuDirectRDMACapable) so that a NIC can
    // read/write the chunks directly, e.g. for GPU-aware MPI between nodes.
    // Throws if a device reports no support for RDMA on VMM allocations.
    bool gpuDirectRDMA = false;
};

class MultiGpuBuffer {
public:
    explicit MultiGpuBuffer(const MultiGpuBufferOptions& options);
    ~MultiGpuBuffer() { destroy(); }

    MultiGpuBuffer(const MultiGpuBuffer&) = delete;
    MultiGpuBuffer& operator=(const MultiGpuBuffer&) = delete;
    MultiGpuBuffer(MultiGpuBuffer&& other) noexcept { swap(other); }
    MultiGpuBuffer& operator=(MultiGpuBuffer&& other) noexcept {
        if (this != &other) {
            destroy();
            swap(other);
        }
        return *this;
    }

    // ---- Pointers ---------------------------------------------------------
    CUdeviceptr devicePtr() const noexcept { return base_; }
    void* data() const noexcept { return reinterpret_cast<void*>(base_); }
    template <typename T>
    T* as() const noexcept { return reinterpret_cast<T*>(base_); }

    // Start of the chunk physically backed by devices()[index].
    void* chunk(int index) const;

    // ---- Geometry ---------------------------------------------------------
    std::size_t size() const noexcept { return size_; }          // total bytes
    std::size_t chunkSize() const noexcept { return chunk_; }    // bytes per GPU
    std::size_t granularity() const noexcept { return granularity_; }
    int numChunks() const noexcept { return static_cast<int>(devices_.size()); }
    const std::vector<int>& devices() const noexcept { return devices_; }
    bool valid() const noexcept { return base_ != 0; }
    explicit operator bool() const noexcept { return valid(); }

    // Which chunk / which GPU backs address p. Returns -1 if p is outside.
    int chunkIndexOf(const void* p) const noexcept;
    int ownerDevice(const void* p) const noexcept;

    // ---- Access control ---------------------------------------------------
    // Set the protection a device sees for the whole range, or for one chunk.
    // `device` may be any CUDA device ordinal (also one that owns no chunk),
    // as long as it can reach the owning GPU(s) via P2P.
    void setAccess(int device, Access access);
    void setAccess(int device, Access access, int chunkIndex);

    // Query the current protection a device has at a given chunk.
    Access getAccess(int device, int chunkIndex) const;

    // True if device a can directly access memory on device b (or a == b).
    static bool peerAccessible(int a, int b);

    // Allocation granularity this class would use for `device` with the
    // given options. Lets callers agree on a common chunk size up front
    // (e.g. across MPI ranks) before constructing a buffer.
    static std::size_t allocationGranularity(int device, bool recommended = true,
                                             bool gpuDirectRDMA = false);

    // The device list a buffer built with `requested` would use:
    // `requested` itself, or all visible devices if it is empty.
    static std::vector<int> resolveDevices(const std::vector<int>& requested);

private:
    void destroy() noexcept;
    void swap(MultiGpuBuffer& other) noexcept;
    void setAccessRange(int device, Access access, CUdeviceptr ptr,
                        std::size_t bytes, int firstChunk, int lastChunk);

    std::vector<int> devices_;                         // CUDA ordinals
    std::vector<CUdevice> cuDevices_;                  // retained so far
    std::vector<CUcontext> contexts_;                  // retained primary ctxs
    std::vector<CUmemGenericAllocationHandle> handles_;
    int mapped_ = 0;                                   // chunks mapped
    CUdeviceptr base_ = 0;
    std::size_t size_ = 0;
    std::size_t chunk_ = 0;
    std::size_t granularity_ = 0;
};

// ===========================================================================
// Implementation details (inline, header-only)
// ===========================================================================
namespace detail {

inline void check(CUresult r, const char* expr, const char* file, int line) {
    if (r == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* desc = nullptr;
    cuGetErrorName(r, &name);
    cuGetErrorString(r, &desc);
    std::ostringstream os;
    os << expr << " failed at " << file << ":" << line << ": "
       << (name ? name : "CUDA_ERROR_?") << " (" << (desc ? desc : "unknown") << ")";
    throw CudaError(r, os.str());
}

inline CUmemAccess_flags toFlags(Access a) {
    switch (a) {
        case Access::None:      return CU_MEM_ACCESS_FLAGS_PROT_NONE;
        case Access::Read:      return CU_MEM_ACCESS_FLAGS_PROT_READ;
        case Access::ReadWrite: return CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }
    return CU_MEM_ACCESS_FLAGS_PROT_NONE;
}

inline Access fromFlags(unsigned long long f) {
    if (f == CU_MEM_ACCESS_FLAGS_PROT_READWRITE) return Access::ReadWrite;
    if (f == CU_MEM_ACCESS_FLAGS_PROT_READ) return Access::Read;
    return Access::None;
}

inline CUmemAllocationProp deviceProp(int ordinal, bool gpuDirectRDMA) {
    CUmemAllocationProp prop = {};
    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = ordinal;
    prop.allocFlags.gpuDirectRDMACapable = gpuDirectRDMA ? 1 : 0;
    // For multi-process sharing, a variant could set
    //   prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    // (or CU_MEM_HANDLE_TYPE_FABRIC for multi-node NVLink) and expose
    // cuMemExportToShareableHandle.
    return prop;
}

// RAII push/pop of a context; throws on push failure.
class ScopedContext {
public:
    explicit ScopedContext(CUcontext ctx) {
        check(cuCtxPushCurrent(ctx), "cuCtxPushCurrent", __FILE__, __LINE__);
    }
    ~ScopedContext() { CUcontext tmp; cuCtxPopCurrent(&tmp); }
    ScopedContext(const ScopedContext&) = delete;
    ScopedContext& operator=(const ScopedContext&) = delete;
};

}  // namespace detail

// Local helper macro; #undef'd at the end of this header.
#define VMM_DETAIL_CHECK(expr) ::vmm::detail::check((expr), #expr, __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
inline MultiGpuBuffer::MultiGpuBuffer(const MultiGpuBufferOptions& opt) {
    if (opt.bytesPerDevice == 0)
        throw std::invalid_argument("MultiGpuBuffer: bytesPerDevice must be > 0");

    devices_ = resolveDevices(opt.devices);
    const int n = static_cast<int>(devices_.size());

    try {
        // 1. Validate devices, retain primary contexts.
        for (int i = 0; i < n; ++i) {
            CUdevice dev;
            VMM_DETAIL_CHECK(cuDeviceGet(&dev, devices_[i]));

            int vmmSupported = 0;
            VMM_DETAIL_CHECK(cuDeviceGetAttribute(
                &vmmSupported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev));
            if (!vmmSupported)
                throw CudaError(CUDA_ERROR_NOT_SUPPORTED,
                                "MultiGpuBuffer: device " + std::to_string(devices_[i]) +
                                    " does not support virtual memory management");

            if (opt.gpuDirectRDMA) {
                int gdr = 0;
                VMM_DETAIL_CHECK(cuDeviceGetAttribute(
                    &gdr, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, dev));
                if (!gdr)
                    throw CudaError(CUDA_ERROR_NOT_SUPPORTED,
                                    "MultiGpuBuffer: device " + std::to_string(devices_[i]) +
                                        " does not support GPUDirect RDMA on VMM allocations");
            }

            CUcontext ctx;
            VMM_DETAIL_CHECK(cuDevicePrimaryCtxRetain(&ctx, dev));
            cuDevices_.push_back(dev);
            contexts_.push_back(ctx);
        }

        // 2. Peer connectivity.
        if (opt.requireFullPeerAccess) {
            for (int a = 0; a < n; ++a)
                for (int b = 0; b < n; ++b)
                    if (a != b && !peerAccessible(devices_[a], devices_[b]))
                        throw CudaError(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
                                        "MultiGpuBuffer: GPU " + std::to_string(devices_[a]) +
                                            " cannot access GPU " + std::to_string(devices_[b]));
        }

        // 3. Granularity common to all devices.
        for (int i = 0; i < n; ++i)
            granularity_ = std::max(
                granularity_, allocationGranularity(devices_[i], opt.useRecommendedGranularity,
                                                    opt.gpuDirectRDMA));
        chunk_ = ((opt.bytesPerDevice + granularity_ - 1) / granularity_) * granularity_;
        size_  = chunk_ * static_cast<std::size_t>(n);

        detail::ScopedContext guard(contexts_[0]);

        // 4. Reserve one contiguous VA range.
        VMM_DETAIL_CHECK(cuMemAddressReserve(&base_, size_, granularity_, 0, 0));

        // 5. Create physical memory on each GPU and map it into the range.
        handles_.reserve(n);
        for (int i = 0; i < n; ++i) {
            CUmemAllocationProp prop = detail::deviceProp(devices_[i], opt.gpuDirectRDMA);
            CUmemGenericAllocationHandle h;
            VMM_DETAIL_CHECK(cuMemCreate(&h, chunk_, &prop, 0));
            handles_.push_back(h);
            VMM_DETAIL_CHECK(cuMemMap(base_ + i * chunk_, chunk_, 0, h, 0));
            ++mapped_;
        }

        // 6. Grant access: each chunk to its owner plus every peer that can reach it.
        for (int i = 0; i < n; ++i) {
            std::vector<CUmemAccessDesc> descs;
            for (int j = 0; j < n; ++j) {
                if (j != i && !peerAccessible(devices_[j], devices_[i])) continue;
                CUmemAccessDesc d = {};
                d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                d.location.id   = devices_[j];
                d.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                descs.push_back(d);
            }
            VMM_DETAIL_CHECK(
                cuMemSetAccess(base_ + i * chunk_, chunk_, descs.data(), descs.size()));
        }
    } catch (...) {
        destroy();
        throw;
    }
}

inline void MultiGpuBuffer::swap(MultiGpuBuffer& o) noexcept {
    std::swap(devices_, o.devices_);
    std::swap(cuDevices_, o.cuDevices_);
    std::swap(contexts_, o.contexts_);
    std::swap(handles_, o.handles_);
    std::swap(mapped_, o.mapped_);
    std::swap(base_, o.base_);
    std::swap(size_, o.size_);
    std::swap(chunk_, o.chunk_);
    std::swap(granularity_, o.granularity_);
}

// Releases whatever has been acquired so far. Safe on partially constructed
// and moved-from objects. Errors are ignored (destructor context).
inline void MultiGpuBuffer::destroy() noexcept {
    bool pushed = false;
    if (!contexts_.empty() && (base_ || !handles_.empty()))
        pushed = (cuCtxPushCurrent(contexts_[0]) == CUDA_SUCCESS);

    for (int i = 0; i < mapped_; ++i) cuMemUnmap(base_ + i * chunk_, chunk_);
    for (auto h : handles_) cuMemRelease(h);
    if (base_) cuMemAddressFree(base_, size_);

    if (pushed) {
        CUcontext tmp;
        cuCtxPopCurrent(&tmp);
    }
    for (auto dev : cuDevices_) cuDevicePrimaryCtxRelease(dev);

    devices_.clear();
    cuDevices_.clear();
    contexts_.clear();
    handles_.clear();
    mapped_ = 0;
    base_ = 0;
    size_ = chunk_ = granularity_ = 0;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
inline void* MultiGpuBuffer::chunk(int index) const {
    if (index < 0 || index >= numChunks())
        throw std::out_of_range("MultiGpuBuffer::chunk: index out of range");
    return reinterpret_cast<void*>(base_ + static_cast<std::size_t>(index) * chunk_);
}

inline int MultiGpuBuffer::chunkIndexOf(const void* p) const noexcept {
    const auto addr = reinterpret_cast<CUdeviceptr>(p);
    if (!base_ || addr < base_ || addr >= base_ + size_) return -1;
    return static_cast<int>((addr - base_) / chunk_);
}

inline int MultiGpuBuffer::ownerDevice(const void* p) const noexcept {
    const int i = chunkIndexOf(p);
    return i < 0 ? -1 : devices_[i];
}

inline bool MultiGpuBuffer::peerAccessible(int a, int b) {
    if (a == b) return true;
    VMM_DETAIL_CHECK(cuInit(0));
    CUdevice da, db;
    VMM_DETAIL_CHECK(cuDeviceGet(&da, a));
    VMM_DETAIL_CHECK(cuDeviceGet(&db, b));
    int ok = 0;
    VMM_DETAIL_CHECK(cuDeviceCanAccessPeer(&ok, da, db));
    return ok != 0;
}

inline std::size_t MultiGpuBuffer::allocationGranularity(int device, bool recommended,
                                                         bool gpuDirectRDMA) {
    VMM_DETAIL_CHECK(cuInit(0));
    CUmemAllocationProp prop = detail::deviceProp(device, gpuDirectRDMA);
    std::size_t g = 0;
    VMM_DETAIL_CHECK(cuMemGetAllocationGranularity(
        &g, &prop,
        recommended ? CU_MEM_ALLOC_GRANULARITY_RECOMMENDED : CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    return g;
}

inline std::vector<int> MultiGpuBuffer::resolveDevices(const std::vector<int>& requested) {
    VMM_DETAIL_CHECK(cuInit(0));
    int count = 0;
    VMM_DETAIL_CHECK(cuDeviceGetCount(&count));
    if (count == 0) throw CudaError(CUDA_ERROR_NO_DEVICE, "MultiGpuBuffer: no CUDA devices");

    std::vector<int> devices = requested;
    if (devices.empty())
        for (int d = 0; d < count; ++d) devices.push_back(d);

    for (int dev : devices) {
        if (dev < 0 || dev >= count)
            throw std::out_of_range("MultiGpuBuffer: invalid device ordinal " +
                                    std::to_string(dev));
        if (std::count(devices.begin(), devices.end(), dev) > 1)
            throw std::invalid_argument("MultiGpuBuffer: duplicate device ordinal " +
                                        std::to_string(dev));
    }
    return devices;
}

// ---------------------------------------------------------------------------
// Access control
// ---------------------------------------------------------------------------
inline void MultiGpuBuffer::setAccess(int device, Access access) {
    setAccessRange(device, access, base_, size_, 0, numChunks() - 1);
}

inline void MultiGpuBuffer::setAccess(int device, Access access, int chunkIndex) {
    if (chunkIndex < 0 || chunkIndex >= numChunks())
        throw std::out_of_range("MultiGpuBuffer::setAccess: chunk index out of range");
    setAccessRange(device, access, base_ + chunkIndex * chunk_, chunk_, chunkIndex, chunkIndex);
}

inline void MultiGpuBuffer::setAccessRange(int device, Access access, CUdeviceptr ptr,
                                           std::size_t bytes, int first, int last) {
    if (!valid()) throw std::logic_error("MultiGpuBuffer: object is empty (moved-from)");

    if (access != Access::None)
        for (int i = first; i <= last; ++i)
            if (!peerAccessible(device, devices_[i]))
                throw CudaError(CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
                                "MultiGpuBuffer::setAccess: GPU " + std::to_string(device) +
                                    " cannot access GPU " + std::to_string(devices_[i]));

    CUmemAccessDesc d = {};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id   = device;
    d.flags         = detail::toFlags(access);

    detail::ScopedContext guard(contexts_[0]);
    VMM_DETAIL_CHECK(cuMemSetAccess(ptr, bytes, &d, 1));
}

inline Access MultiGpuBuffer::getAccess(int device, int chunkIndex) const {
    if (!valid()) throw std::logic_error("MultiGpuBuffer: object is empty (moved-from)");
    if (chunkIndex < 0 || chunkIndex >= numChunks())
        throw std::out_of_range("MultiGpuBuffer::getAccess: chunk index out of range");
    CUmemLocation loc = {};
    loc.type = CU_MEM_LOCATION_TYPE_DEVICE;
    loc.id   = device;
    unsigned long long flags = 0;
    detail::ScopedContext guard(contexts_[0]);
    VMM_DETAIL_CHECK(cuMemGetAccess(&flags, &loc, base_ + chunkIndex * chunk_));
    return detail::fromFlags(flags);
}

#undef VMM_DETAIL_CHECK

}  // namespace vmm
