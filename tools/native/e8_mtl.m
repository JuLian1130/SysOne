// E8b: Metal host for the GPU half of the prefill engine.
//
// The whole point of E8b is that an ANE output IOSurface is handed to the GPU WITHOUT A
// COPY. E4b proved `newBufferWithBytesNoCopy` can wrap IOSurface memory (1/4/16/64 MB,
// bytes verified through a blit); this file turns that into the engine's GPU backend.
//
// Every ANE surface in the E8 pipeline is channel-major: a (C, S) fp16 tensor is stored as
// x[c * stride + s].  The kernels therefore all take absolute column offsets, so a
// kernel can address a tile of a full-length buffer or a standalone tile buffer with the
// same code.
//
// C ABI (see e8_mtl.py for the ctypes side):
//   e8m_init(metallibPath)                       -> 0 / 1
//   e8m_wrap(surfaceHandle)                      -> bufferHandle   (zero copy, cached)
//   e8m_alloc(bytes)                             -> bufferHandle   (shared, CPU-writable)
//   e8m_base(bufferHandle)                       -> void*          (CPU view)
//   e8m_bytes(bufferHandle)                      -> uint64
//   e8m_free(bufferHandle)
//   e8m_dispatch(name, bufs[], nbufs, args, argsLen, grid[3], tg[3]) -> 0 / 1
//   e8m_commit()                                 -> 0 / 1   (async)
//   e8m_wait()                                   -> wall ms of the last commit
//   e8m_sync()                                   -> commit+wait
//   e8m_gpu_ms()                                 -> GPU time of the last waited CB
//   e8m_pipe_max_threads(name)                   -> maxTotalThreadsPerThreadgroup
//   e8m_tgmem_bytes()                            -> maxThreadgroupMemoryLength
//   e8m_last_error()
//
// Build (see build_e8.sh):
//   xcrun -sdk macosx metal -c e8_kernels.metal -I. -o /tmp/e8/k1.air      (per kernel file)
//   xcrun -sdk macosx metallib /tmp/e8/*.air -o /tmp/e8/e8.metallib
//   clang -dynamiclib -fobjc-arc -O2 -framework Foundation -framework Metal \
//         -framework IOSurface -o /tmp/e8/libe8mtl.dylib e8_mtl.m

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <IOSurface/IOSurface.h>
#include <mach/mach_time.h>
#include <string.h>
#include <stdio.h>

static mach_timebase_info_data_t g_tb;
static id<MTLDevice> g_dev;
static id<MTLLibrary> g_lib;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipes;
static NSMutableDictionary<NSNumber *, id<MTLBuffer>> *g_bufs;
static NSMutableDictionary<NSNumber *, NSNumber *> *g_surf2h;
static uint64_t g_next = 1;
// Two independent command queues, so the tiny kernels (norm/norm2/copy) no longer have to
// wait behind a 140 ms attention dispatch.
//
//   lane 0  long GPU stages: k_attn, k_gdn, the gathers -- the stream the ANE hides behind
//   lane 1  the 1-3 ms kernels: k_norm_to, k_norm_resid, k_copy
//
// Why this exists: with ONE in-order queue the per-tile chain is
//   GPU attn(s) -> ANE out(s) -> GPU norm2(s) -> ANE ffn(s) -> GPU attn(s+1)
// and norm2(s) can only be encoded after out(s) returns, so it is queued *behind* attn(s+1)
// and the ANE's next ANE stage stalls on it.  Measured: 84 ms/layer of GPU idle in the
// attention layers.  On its own queue norm2(s) runs during attn(s+1) and the stall is gone.
//
// Inside a lane, `cb`/`enc` is the OPEN command buffer and `pending` holds the committed but
// not yet waited ones.  They are separate because a bare commit() must not swallow the next
// dispatch (that bug cost the whole A2 wavefront its correctness: a dispatch encoded into a
// nil encoder vanished silently, and the following sync no-op'd too).
typedef struct {
    id<MTLCommandQueue> q;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;
    NSMutableArray<id<MTLCommandBuffer>> *pending;
    double busy_ms;
} Lane;
#define E8_NLANE 2   // attention+long, small kernels (see e8_a3.py on why not 3)
static Lane g_lanes[E8_NLANE];
static char g_err[1024];
static double g_last_wall_ms, g_last_gpu_ms;
// E8M_BARRIER: put a buffers-scope memory barrier in front of every dispatch.  Metal orders the
// dispatches of one compute encoder, but ordering is not visibility: without a barrier the next
// dispatch may read its predecessor's writes through the cache.  A single-dispatch stage never
// cares, a multi-dispatch stage does -- and the fast GDN core is three dispatches deep over one
// shared scratch buffer (k_gdn_prep -> k_gdn -> k_gdn_norm).  Default off: measure it first.
static int g_barrier = 0;


static void setErr(NSString *s) {
    snprintf(g_err, sizeof(g_err), "%s", s ? s.UTF8String : "(nil)");
}
static double ms_since(uint64_t t0) {
    return (double)(mach_absolute_time() - t0) * g_tb.numer / g_tb.denom / 1e6;
}

// Drop whatever a *previous* engine left on the lanes.  The lanes are process-global (one queue
// per lane, shared by every Engine), but each Engine's command-buffer ids start at 1 again, so a
// leftover makes the next engine reap someone else's buffer and mark its own work done early --
// i.e. a follower layer reads a half-written h.  Measured without this call: an A3 run under an
// A1m run in the same process gives max|d| up to 2.98 at nseg=16, while A3 in a fresh process is
// bit-exact 4/4 against a file-based A1m reference (plan v2, S1 follow-up).
int e8m_reset_lanes(void) {
    @autoreleasepool {
        for (int i = 0; i < E8_NLANE; i++) {
            Lane *L = &g_lanes[i];
            if (L->enc) { [L->enc endEncoding]; L->enc = nil; }
            if (L->cb) { [L->cb commit]; L->cb = nil; }
            if (L->pending) {
                for (id<MTLCommandBuffer> cb in L->pending) [cb waitUntilCompleted];
                [L->pending removeAllObjects];
            }
            L->busy_ms = 0.0;
        }
        return 0;
    }
}

static int make_queues(void) {
    for (int i = 0; i < E8_NLANE; i++) {
        if (!g_lanes[i].q) g_lanes[i].q = [g_dev newCommandQueue];
        if (!g_lanes[i].q) { setErr(@"newCommandQueue failed"); return 1; }
        if (!g_lanes[i].pending) g_lanes[i].pending = [NSMutableArray array];
    }
    return 0;
}

int e8m_init(const char *metallibPath) {
    const char *bar = getenv("E8M_BARRIER");
    g_barrier = (bar && bar[0] == '1') ? 1 : 0;
    @autoreleasepool {
        if (g_dev) return 0;
        mach_timebase_info(&g_tb);
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { setErr(@"no Metal device"); return 1; }
        if (make_queues()) return 6;
        NSError *e = nil;
        g_lib = [g_dev newLibraryWithURL:[NSURL fileURLWithPath:
                    [NSString stringWithUTF8String:metallibPath]] error:&e];
        if (!g_lib) {
            setErr([NSString stringWithFormat:@"newLibraryWithURL: %@", e.localizedDescription]);
            return 2;
        }
        g_pipes = [NSMutableDictionary dictionary];
        g_bufs = [NSMutableDictionary dictionary];
        g_surf2h = [NSMutableDictionary dictionary];
        fprintf(stderr, "[e8m] %s | %lu functions | tgmem %lu | maxBuf %.2f GB\n",
                g_dev.name.UTF8String, (unsigned long)g_lib.functionNames.count,
                (unsigned long)g_dev.maxThreadgroupMemoryLength,
                g_dev.maxBufferLength / 1e9);
        return 0;
    }
}

// Compile from .metal SOURCE at startup.  This machine has only the Command Line Tools (no
// Xcode), so `xcrun metal` does not exist -- but the runtime compiler in the Metal framework
// does, which makes the build step a runtime step.  The first source file must be the common
// header; its `#include "e8_common.h"` lines in the kernel files are dropped because the
// header is already prepended (runtime compilation has no include path).
int e8m_init_src(const char *const *paths, int npaths, int verbose) {
    @autoreleasepool {
        if (g_dev) return 0;
        mach_timebase_info(&g_tb);
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) { setErr(@"no Metal device"); return 1; }
        if (make_queues()) return 6;

        NSMutableString *src = [NSMutableString string];
        for (int i = 0; i < npaths; i++) {
            NSString *p = [NSString stringWithUTF8String:paths[i]];
            NSError *fe = nil;
            NSString *t = [NSString stringWithContentsOfFile:p encoding:NSUTF8StringEncoding
                                                       error:&fe];
            if (!t) {
                setErr([NSString stringWithFormat:@"read %@: %@", p, fe.localizedDescription]);
                return 3;
            }
            if ([p hasSuffix:@".metal"]) {
                NSMutableArray<NSString *> *keep = [NSMutableArray array];
                for (NSString *line in [t componentsSeparatedByString:@"\n"]) {
                    if ([line containsString:@"#include \"e8_common.h\""]) continue;
                    [keep addObject:line];
                }
                t = [keep componentsJoinedByString:@"\n"];
            }
            [src appendString:t];
            [src appendString:@"\n"];
        }
        [src writeToFile:@"/tmp/e8/e8_source.metal" atomically:YES
                encoding:NSUTF8StringEncoding error:nil];

        uint64_t t0 = mach_absolute_time();
        NSError *e = nil;
        MTLCompileOptions *opt = [MTLCompileOptions new];
        g_lib = [g_dev newLibraryWithSource:src options:opt error:&e];
        if (!g_lib) {
            setErr([NSString stringWithFormat:@"newLibraryWithSource: %@",
                    e.localizedDescription ?: @"(no message)"]);
            return 4;
        }
        g_pipes = [NSMutableDictionary dictionary];
        g_bufs = [NSMutableDictionary dictionary];
        g_surf2h = [NSMutableDictionary dictionary];
        if (verbose) {
            fprintf(stderr, "[e8m] %s | compiled %d source files (%lu chars) in %.0f ms | "
                    "%lu functions | tgmem %lu\n",
                    g_dev.name.UTF8String, npaths, (unsigned long)src.length,
                    ms_since(t0), (unsigned long)g_lib.functionNames.count,
                    (unsigned long)g_dev.maxThreadgroupMemoryLength);
        }
        return 0;
    }
}

const char *e8m_last_error(void) { return g_err; }
uint64_t e8m_tgmem_bytes(void) { return g_dev ? g_dev.maxThreadgroupMemoryLength : 0; }
uint64_t e8m_max_threads_per_tg(void) {
    return g_dev ? g_dev.maxThreadsPerThreadgroup.width : 0;
}

static id<MTLComputePipelineState> pipe_for(NSString *name) {
    id<MTLComputePipelineState> p = g_pipes[name];
    if (p) return p;
    id<MTLFunction> f = [g_lib newFunctionWithName:name];
    if (!f) { setErr([NSString stringWithFormat:@"no kernel named %@", name]); return nil; }
    NSError *e = nil;
    p = [g_dev newComputePipelineStateWithFunction:f error:&e];
    if (!p) { setErr([NSString stringWithFormat:@"pipeline %@: %@", name, e.localizedDescription]); return nil; }
    g_pipes[name] = p;
    return p;
}

uint64_t e8m_pipe_max_threads(const char *name) {
    @autoreleasepool {
        id<MTLComputePipelineState> p = pipe_for([NSString stringWithUTF8String:name]);
        return p ? p.maxTotalThreadsPerThreadgroup : 0;
    }
}

// ---------------------------------------------------------------- buffers
uint64_t e8m_wrap(uint64_t surf) {
    @autoreleasepool {
        if (!g_dev) { setErr(@"e8m_wrap before init"); return 0; }
        NSNumber *k = @(surf);
        NSNumber *h = g_surf2h[k];
        IOSurfaceRef s = (IOSurfaceRef)surf;
        size_t alloc = IOSurfaceGetAllocSize(s);
        if (h) {
            // The cache is keyed by the IOSurface pointer and IOSurfaceCreate hands a released
            // pointer straight back to the next allocation, so a stale entry here would return
            // a buffer over *freed* memory (A3's cross-block surface free did exactly that and
            // corrupted whole blocks).  e8m_unwrap() is the real fix; this check is the net.
            id<MTLBuffer> old = g_bufs[h];
            if (old && old.length == alloc) return h.unsignedLongLongValue;
            [g_surf2h removeObjectForKey:k];
            [g_bufs removeObjectForKey:h];
            setErr(@"wrap: stale surface handle, re-wrapping");
            h = nil;
        }

        IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
        void *base = IOSurfaceGetBaseAddress(s);
        size_t alloc2 = IOSurfaceGetAllocSize(s);
        IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
        if (!base || !alloc2) { setErr(@"wrap: no base/alloc"); return 0; }
        alloc = alloc2;

        id<MTLBuffer> b = nil;
        @try {
            b = [g_dev newBufferWithBytesNoCopy:base length:alloc
                                        options:MTLResourceStorageModeShared deallocator:nil];
        } @catch (NSException *e) { b = nil; }
        if (!b) {
            // fall back to an explicit copy so a driver that refuses the wrap is diagnosable
            // instead of fatal; the caller can see it in the error string.
            b = [g_dev newBufferWithLength:alloc options:MTLResourceStorageModeShared];
            if (!b) { setErr(@"wrap: newBufferWithBytesNoCopy refused and copy alloc failed"); return 0; }
            memcpy(b.contents, base, alloc);
            setErr(@"wrap: FELL BACK TO COPY (zero-copy refused)");
        }
        uint64_t hh = g_next++;
        g_bufs[@(hh)] = b;
        g_surf2h[k] = @(hh);
        return hh;
    }
}

// Drop the wrapper for a surface that is about to be released.  Without this the cache above
// keeps a buffer over memory the allocator is free to hand to a completely different surface.
int e8m_unwrap(uint64_t surf) {
    @autoreleasepool {
        if (!surf) return 0;
        NSNumber *k = @(surf);
        NSNumber *h = g_surf2h[k];
        if (!h) return 0;
        [g_surf2h removeObjectForKey:k];
        [g_bufs removeObjectForKey:h];
        return 1;
    }
}

uint64_t e8m_alloc(uint64_t bytes) {    @autoreleasepool {
        if (!g_dev) { setErr(@"e8m_alloc before init"); return 0; }
        id<MTLBuffer> b = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!b) { setErr(@"e8m_alloc failed"); return 0; }
        uint64_t h = g_next++;
        g_bufs[@(h)] = b;
        return h;
    }
}

void *e8m_base(uint64_t h) {
    id<MTLBuffer> b = g_bufs[@(h)];
    return b ? b.contents : NULL;
}
uint64_t e8m_bytes(uint64_t h) {
    id<MTLBuffer> b = g_bufs[@(h)];
    return b ? (uint64_t)b.length : 0;
}
void e8m_free(uint64_t h) {
    @autoreleasepool {
        NSNumber *k = @(h);
        id<MTLBuffer> b = g_bufs[k];
        if (!b) return;
        for (NSNumber *s in g_surf2h) {
            if (g_surf2h[s].unsignedLongLongValue == h) { [g_surf2h removeObjectForKey:s]; break; }
        }
        [g_bufs removeObjectForKey:k];
    }
}

// ---------------------------------------------------------------- dispatch
// args go to buffer index 30 (E8_ARGS_INDEX in e8_common.h).
// `lane` selects the command queue (0 = long stages, 1 = small kernels); see the Lane comment.
int e8m_dispatch_l(int lane, const char *name, const uint64_t *bufs, int nbufs,
                   const void *args, int argsLen,
                   uint32_t gx, uint32_t gy, uint32_t gz,
                   uint32_t tgx, uint32_t tgy, uint32_t tgz) {
    @autoreleasepool {
        if (lane < 0 || lane >= E8_NLANE) { setErr(@"lane out of range"); return 5; }
        Lane *L = &g_lanes[lane];
        id<MTLComputePipelineState> p = pipe_for([NSString stringWithUTF8String:name]);
        if (!p) return 1;
        int had_dispatch = (L->enc != nil);
        if (!L->enc) {
            L->cb = [L->q commandBuffer];
            L->enc = [L->cb computeCommandEncoder];
            if (!L->enc) { setErr(@"computeCommandEncoder nil"); return 2; }
        }
        if (g_barrier && had_dispatch) {
            [L->enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }
        [L->enc setComputePipelineState:p];
        for (int i = 0; i < nbufs; i++) {
            id<MTLBuffer> b = g_bufs[@(bufs[i])];
            if (!b) { setErr([NSString stringWithFormat:@"dispatch %s: bad buffer handle %llu",
                              name, (unsigned long long)bufs[i]]); return 3; }
            [L->enc setBuffer:b offset:0 atIndex:i];
        }
        if (args && argsLen > 0) {
            if (argsLen > 4096) { setErr(@"args > 4096 bytes"); return 4; }
            [L->enc setBytes:args length:(NSUInteger)argsLen atIndex:30];
        }
        [L->enc dispatchThreadgroups:MTLSizeMake(gx ? gx : 1, gy ? gy : 1, gz ? gz : 1)
              threadsPerThreadgroup:MTLSizeMake(tgx, tgy, tgz)];
        return 0;
    }
}

int e8m_dispatch(const char *name, const uint64_t *bufs, int nbufs,
                 const void *args, int argsLen,
                 uint32_t gx, uint32_t gy, uint32_t gz,
                 uint32_t tgx, uint32_t tgy, uint32_t tgz) {
    return e8m_dispatch_l(0, name, bufs, nbufs, args, argsLen, gx, gy, gz, tgx, tgy, tgz);
}

int e8m_commit_l(int lane) {
    @autoreleasepool {
        if (lane < 0 || lane >= E8_NLANE) { setErr(@"lane out of range"); return 5; }
        Lane *L = &g_lanes[lane];
        // nothing encoded since the last commit -> no-op (committing twice is an assertion
        // failure inside IOGPUMetalCommandBuffer)
        if (!L->cb || !L->enc) return 0;
        [L->enc endEncoding];
        [L->cb commit];
        L->enc = nil;
        [L->pending addObject:L->cb];
        L->cb = nil;
        return 0;
    }
}

int e8m_commit(void) { return e8m_commit_l(0); }

// Wait for the first `k` pending buffers (k < 0 = all of them) and return the ms spent
// blocked.  Partial waits are what makes a deep wavefront possible: the ANE must be able to
// reap attn(s) while attn(s+1) keeps running.
double e8m_wait_upto(int lane, int k) {
    @autoreleasepool {
        if (lane < 0 || lane >= E8_NLANE) { setErr(@"lane out of range"); return -1; }
        Lane *L = &g_lanes[lane];
        if (e8m_commit_l(lane)) return -1;
        NSUInteger n = L->pending.count;
        if (k < 0 || (NSUInteger)k > n) k = (int)n;
        if (n == 0) return 0.0;
        uint64_t t0 = mach_absolute_time();
        double gpu = 0.0;
        for (NSInteger i = 0; i < (NSInteger)k; i++) {
            id<MTLCommandBuffer> cb = L->pending[(NSUInteger)i];
            [cb waitUntilCompleted];
            gpu += (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
            if (cb.error) setErr([NSString stringWithFormat:@"CB error: %@", cb.error.localizedDescription]);
        }
        g_last_wall_ms = ms_since(t0);
        g_last_gpu_ms = gpu;
        L->busy_ms += gpu;
        if ((NSInteger)k >= (NSInteger)n) [L->pending removeAllObjects];
        else for (NSInteger i = (NSInteger)k - 1; i >= 0; i--) [L->pending removeObjectAtIndex:(NSUInteger)i];
        return g_last_wall_ms;
    }
}

double e8m_wait_l(int lane) { return e8m_wait_upto(lane, -1); }
double e8m_wait(void) { return e8m_wait_l(0); }

double e8m_sync_l(int lane) {
    if (e8m_commit_l(lane)) return -1;
    return e8m_wait_upto(lane, -1);
}
double e8m_sync(void) { return e8m_sync_l(0); }

double e8m_gpu_ms(void) { return g_last_gpu_ms; }
// Total time the GPU was executing this lane's command buffers.  lane 0 busy vs the ANE total
// is what says whether the GPU or the ANE is the wall; lane 1 is concurrent with lane 0 and
// therefore overlaps in wall time.
double e8m_gpu_busy_l(int lane) {
    if (lane < 0 || lane >= E8_NLANE) return -1;
    double v = g_lanes[lane].busy_ms;
    g_lanes[lane].busy_ms = 0.0;
    return v;
}
double e8m_gpu_busy_ms(void) { return e8m_gpu_busy_l(0); }
double e8m_last_wall_ms(void) { return g_last_wall_ms; }
