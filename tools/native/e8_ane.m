// e8_ane.m — E8 prototype: ANE backend as a plain C-ABI shared library.
//
// Why a dylib: the E8 A1 prototype has to run the ANE projection stages and the MLX
// attention/GDN stages IN ONE PROCESS so that the wall clock and the wired-memory peak are
// real numbers rather than a sum of two separate experiments. MLX is only usable from
// Python, and Python cannot talk to the private _ANE* ObjC classes — but it can dlopen a
// dylib. So this file keeps ONLY the private-API plumbing (compile / load / attach / eval)
// and nothing about the model: MIL text and weight blobs are built in Python
// (e8_ane.py) and handed over as (utf8 string, file path).
//
// Derived from ane_mil_probe.m (same private-API dialect: _ANEInMemoryModelDescriptor ->
// _ANEInMemoryModel -> compileWithQoS:/loadWithQoS: -> _ANERequest/_ANEIOSurfaceObject ->
// evaluateWithQoS:). Differences:
//   * IOSurfaces are created by the CALLER (Python) and passed in as uint64 handles, so the
//     Python side can wrap them zero-copy with numpy and account for their memory itself;
//   * one program can hold several procedures (func blocks); requests are kept per
//     procedureIndex instead of being torn down and rebuilt;
//   * weight blobs are read from a file with NSDataReadingMappedIfSafe, so a 20-layer
//     residency test does not keep 2.3 GB of dirty anonymous memory on the host side
//     (this was one of the two causes of the kernel panics recorded in
//     ../e1a-e1b-results.md §7.4);
//   * eval returns its own mach_absolute_time delta in ms so Python timing excludes ctypes.
//
// Build:
//   clang -dynamiclib -fobjc-arc -O2 -ffp-contract=off \
//         -framework Foundation -framework IOSurface \
//         -o /tmp/e8/libe8ane.dylib e8_ane.m
//
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach.h>
#import <mach/mach_time.h>
#import <objc/message.h>
#import <objc/runtime.h>
#import <dlfcn.h>
#include <string.h>
#include <stdlib.h>

// How many (procedure, slot) bindings one loaded program may hold.  The vision tower uses one
// slot per token tile, so 8 was not enough once the bucket drops to 256.
#define SLOTS_PER_PROC 64

@interface E8Prog : NSObject
@property(nonatomic, strong) id model;
@property(nonatomic, strong) NSMutableDictionary *reqs;   // @(proc) -> _ANERequest
@property(nonatomic, strong) NSMutableDictionary *ios;    // @(proc) -> @[NSValue(ioIn), NSValue(ioOut)]
// Slot 0 is the plain single-buffered path.  Slots 1..n exist so the A2 wavefront can
// ping-pong two surface sets without re-creating requests: creating a request per tile
// would leak a CFRetain on each surface (attach retains, nothing releases the old pair).
@property(nonatomic, strong) NSMutableDictionary *sreqs;  // @(proc*8+slot) -> _ANERequest
@property(nonatomic, strong) NSMutableDictionary *sios;   // @(proc*8+slot) -> @[in, out]
@property(nonatomic, strong) NSString *tmpDir;
@property(nonatomic) double compileMs;   // time inside compileWithQoS:
@property(nonatomic) double loadMs;      // time inside loadWithQoS:
@property(nonatomic) int cacheHit;       // aned already held the lowered program
@property(nonatomic) int suspended;         // device residency dropped, object kept
@property(nonatomic) int resumes;           // how many e8_resume calls succeeded
@property(nonatomic) double resumeMs;       // time inside the last bare loadWithQoS:
@end

@implementation E8Prog
@end

static Class KDesc, KIMM, KReq, KIO;
static NSMutableDictionary<NSString *, E8Prog *> *g_progs;
static char g_err[4096];
static double g_nsPerTick = 0;

// ---------------------------------------------------------------- threading
//
// eval and load are meant to run on DIFFERENT threads: the layer-outer streaming design
// prefetches layer L+1 while the ANE evaluates layer L, so "can a warm load overlap an eval"
// is a measured question (e3b_aned_overlap.py), not something to assume.  Everything that
// could make that false is kept out of this file: an E8Prog is immutable after construction
// except for its own request table, which only its owner thread touches.  The one piece of
// shared mutable state is the registry below, and the lock guards only the dictionary
// access -- never an ANE call -- so it cannot serialise anything that matters.
static NSLock *g_lock;

static E8Prog *progGet(const char *name) {
    if (!name) return nil;
    [g_lock lock];
    E8Prog *p = g_progs[[NSString stringWithUTF8String:name]];
    [g_lock unlock];
    return p;
}

static void setErr(NSString *s) {
    snprintf(g_err, sizeof g_err, "%s", s ? s.UTF8String : "(nil)");
}
const char *e8_last_error(void) { return g_err; }

/* The compiler's real reason lives one level down: localizedDescription is only
   "_ANECompiler : ANECCompile() FAILED", while NSUnderlyingError says err=(InvalidMILProgram)
   or the actual parse complaint.  Walk the chain so probe failures are diagnosable. */
static NSString *e8ErrText(NSError *e) {
    NSMutableString *s = [NSMutableString string];
    for (int depth = 0; e && depth < 5; depth++) {
        [s appendFormat:@"%@|%@|", [e domain], [e localizedDescription]];
        e = [e userInfo][NSUnderlyingErrorKey];
    }
    return s;
}

static void ensureTimebase(void) {
    if (g_nsPerTick > 0) return;
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    g_nsPerTick = (double)tb.numer / (double)tb.denom;
}

int e8_init(void) {
    @autoreleasepool {
        if (KDesc) return 0;
        void *h = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",
                         RTLD_NOW);
        if (!h) { setErr(@"dlopen AppleNeuralEngine failed"); return 1; }
        KDesc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        KIMM  = NSClassFromString(@"_ANEInMemoryModel");
        KReq  = NSClassFromString(@"_ANERequest");
        KIO   = NSClassFromString(@"_ANEIOSurfaceObject");
        if (!KDesc || !KIMM || !KReq || !KIO) {
            setErr(@"private ANE classes missing (macOS too new / class renamed?)");
            return 2;
        }
        if (!g_progs) g_progs = [NSMutableDictionary dictionary];
        if (!g_lock) g_lock = [NSLock new];
        ensureTimebase();
        g_err[0] = 0;
    }
    return 0;
}

// ---------------------------------------------------------------- compile + load (weights resident)
//
// 2026-10-30: `_ANEInMemoryModel` exposes a private selector `compiledModelExists`.  aned keeps
// the *lowered* ANE program keyed by the descriptor's content hash (hexStringIdentifier), and
// `loadWithQoS:` alone reuses that lowering.  Calling `compileWithQoS:` again on a hit is NOT
// free -- it re-runs the whole lowering, which is exactly the 580-900 ms/layer measured in
// ../e3-attention-results.md §10.7.  So:
//
//   e8_load        always compiles           (historical behaviour; keeps old numbers comparable)
//   e8_load_warm   skips compile on a hit    (what a warm process should do)
//   e8_compiled_exists  peek the cache without loading anything
//
// The cache lives in aned (daemon state): it survives across processes but not across an aned
// restart, and its eviction policy is opaque.

static int e8_load_impl(const char *name, const char *milUtf8, const char *blobPath, int useCache) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return 1;
        NSString *key = [NSString stringWithUTF8String:name];
        if (!key || !milUtf8 || !blobPath) { setErr(@"e8_load: null argument"); return 2; }
        [g_lock lock];
        BOOL dup = g_progs[key] != nil;
        [g_lock unlock];
        if (dup) { setErr([NSString stringWithFormat:@"e8_load: '%@' already loaded", key]); return 3; }

        NSError *e = nil;
        NSData *milData = [[NSString stringWithUTF8String:milUtf8] dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blobPath]
                                             options:NSDataReadingMappedIfSafe error:&e];
        if (!blob) {
            setErr([NSString stringWithFormat:@"blob read failed: %@", [e localizedDescription]]);
            return 4;
        }

        NSDictionary *wdict = @{@"@model_path/weights.bin": @{@"offset": @0, @"data": blob}};
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            KDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!desc) { setErr(@"modelWithMILText -> nil (bad MIL?)"); return 5; }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(KIMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { setErr(@"inMemoryModelWithDescriptor -> nil"); return 6; }

        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];

        SEL existsSel = sel_getUid("compiledModelExists");
        BOOL cacheHit = NO;
        if (useCache && [model respondsToSelector:existsSel]) {
            cacheHit = ((BOOL(*)(id,SEL))objc_msgSend)(model, existsSel);
        }

        double compileMs = 0, loadMs = 0;
        if (!cacheHit) {
            [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
          withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [blob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights.bin"] atomically:YES];

            uint64_t t0 = mach_absolute_time();
            BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            compileMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
            if (!ok) {
                setErr([NSString stringWithFormat:@"compile FAILED: %@", e8ErrText(e)]);
                [fm removeItemAtPath:tmpDir error:nil];
                return 7;
            }
        }

        uint64_t t1 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        loadMs = (double)(mach_absolute_time() - t1) * g_nsPerTick / 1e6;
        if (!ok) {
            setErr([NSString stringWithFormat:@"load FAILED: %@", e8ErrText(e)]);
            [fm removeItemAtPath:tmpDir error:nil];
            return 8;
        }

        E8Prog *p = [E8Prog new];
        p.model = model;
        p.tmpDir = tmpDir;
        p.compileMs = compileMs;
        p.loadMs = loadMs;
        p.cacheHit = cacheHit ? 1 : 0;
        p.reqs = [NSMutableDictionary dictionary];
        p.ios = [NSMutableDictionary dictionary];
        p.sreqs = [NSMutableDictionary dictionary];
        p.sios = [NSMutableDictionary dictionary];
        [g_lock lock];
        g_progs[key] = p;
        [g_lock unlock];
        g_err[0] = 0;
    }
    return 0;
}

// Historical entry point: always compiles.
int e8_load(const char *name, const char *milUtf8, const char *blobPath) {
    return e8_load_impl(name, milUtf8, blobPath, 0);
}

// Compile-cache aware entry point: on an aned hit, call only loadWithQoS:.
int e8_load_warm(const char *name, const char *milUtf8, const char *blobPath) {
    return e8_load_impl(name, milUtf8, blobPath, 1);
}

// Peek without loading: does aned already hold the lowered program for this content?
// 1 = yes, 0 = no, -1 = the private selector is missing on this OS, -2 = error.
int e8_compiled_exists(const char *milUtf8, const char *blobPath) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return -2;
        NSError *e = nil;
        NSData *milData = [[NSString stringWithUTF8String:milUtf8] dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blobPath]
                                             options:NSDataReadingMappedIfSafe error:&e];
        if (!blob) { setErr(@"e8_compiled_exists: blob read failed"); return -2; }
        NSDictionary *wdict = @{@"@model_path/weights.bin": @{@"offset": @0, @"data": blob}};
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            KDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!desc) { setErr(@"e8_compiled_exists: descriptor -> nil"); return -2; }
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(KIMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { setErr(@"e8_compiled_exists: model -> nil"); return -2; }
        SEL sel = sel_getUid("compiledModelExists");
        if (![model respondsToSelector:sel]) return -1;
        return ((BOOL(*)(id,SEL))objc_msgSend)(model, sel) ? 1 : 0;
    }
}

// `loadWithQoS:` costs 26.5 ms when called straight after `load_fast`, but 4.6 ms inside the
// concurrency probe -- where a `compiled_exists` peek runs immediately before every load.
// That is a 5.9x difference in the same API on the same machine, and it decides whether a
// layer-outer prefetch costs 27 ms/layer or 5 ms/layer (i.e. where the "full coverage" threshold
// sits).  So take the peek apart: it is three separate sub-steps (build the descriptor =
// hash 117 MB; build the `_ANEInMemoryModel`; ask aned `compiledModelExists`), and any one of
// them could be what warms the later load.
//
//   level 0  do nothing at all (control)
//   level 1  build the descriptor only
//   level 2  + build the model object, then let it go
//   level 3  + `compiledModelExists` query  (== what e8_compiled_exists does)
//   level 4  like 3 but the model object is KEPT alive in g_peekHold (does holding it -- and
//            therefore its device-side residency -- explain the cheap load?)
//
// Timings come back per sub-step.  Returns 0 ok | -1 selector missing | -2 error.
static NSMutableArray *g_peekHold;

int e8_peek(const char *milUtf8, const char *blobPath, int level,
            double *descMs, double *modelMs, double *queryMs, int *exists) {
    @autoreleasepool {
        if (descMs) *descMs = 0;
        if (modelMs) *modelMs = 0;
        if (queryMs) *queryMs = 0;
        if (exists) *exists = -1;
        if (level <= 0) return 0;
        if (!KDesc && e8_init() != 0) return -2;
        NSError *e = nil;
        NSData *milData = [[NSString stringWithUTF8String:milUtf8]
                              dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blobPath]
                                             options:NSDataReadingMappedIfSafe error:&e];
        if (!blob) { setErr(@"e8_peek: blob read failed"); return -2; }
        NSDictionary *wdict = @{@"@model_path/weights.bin": @{@"offset": @0, @"data": blob}};

        uint64_t t0 = mach_absolute_time();
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            KDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (descMs) *descMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!desc) { setErr(@"e8_peek: descriptor -> nil"); return -2; }
        if (level == 1) return 0;

        t0 = mach_absolute_time();
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(KIMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (modelMs) *modelMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!model) { setErr(@"e8_peek: model -> nil"); return -2; }
        if (level == 2) return 0;

        t0 = mach_absolute_time();
        SEL sel = sel_getUid("compiledModelExists");
        BOOL ok = [model respondsToSelector:sel]
                    ? ((BOOL(*)(id,SEL))objc_msgSend)(model, sel) : NO;
        if (queryMs) *queryMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (exists) *exists = ok ? 1 : 0;
        if (![model respondsToSelector:sel]) return -1;
        if (level >= 4) {
            if (!g_peekHold) g_peekHold = [NSMutableArray array];
            [g_peekHold addObject:model];      // deliberately not released
        }
        g_err[0] = 0;
        return 0;
    }
}

int e8_peek_hold_count(void) { return (int)g_peekHold.count; }

void e8_peek_hold_clear(void) { g_peekHold = nil; }

// compile_ms / load_ms / cache_hit for one program.  0 on success.
int e8_stats(const char *name, double *compileMs, double *loadMs, int *cacheHit) {
    E8Prog *p = g_progs[[NSString stringWithUTF8String:name]];
    if (!p) { setErr(@"e8_stats: unknown program"); return 1; }
    if (compileMs) *compileMs = p.compileMs;
    if (loadMs) *loadMs = p.loadMs;
    if (cacheHit) *cacheHit = p.cacheHit;
    return 0;
}

// ---------------------------------------------------------------- IOSurfaces (owned by the caller)

uint64_t e8_surface_new(uint64_t bytes) {
    @autoreleasepool {
        NSDictionary *d = @{(id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0};
        IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)d);
        return (uint64_t)s;
    }
}

void *e8_surface_base(uint64_t surf) {
    if (!surf) { setErr(@"e8_surface_base: null"); return NULL; }
    IOSurfaceLock((IOSurfaceRef)surf, kIOSurfaceLockReadOnly, NULL);
    void *base = IOSurfaceGetBaseAddress((IOSurfaceRef)surf);
    IOSurfaceUnlock((IOSurfaceRef)surf, kIOSurfaceLockReadOnly, NULL);
    return base;
}

uint64_t e8_surface_bytes(uint64_t surf) {
    if (!surf) return 0;
    return (uint64_t)IOSurfaceGetAllocSize((IOSurfaceRef)surf);
}

void e8_surface_free(uint64_t surf) {
    if (!surf) return;
    CFRelease((CFTypeRef)surf);
}

// ---------------------------------------------------------------- request binding per procedure

int e8_attach_slot(const char *name, int proc, int slot, uint64_t inSurf, uint64_t outSurf) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return 1;
        E8Prog *p = progGet(name);
        if (!p) { setErr(@"e8_attach: unknown program"); return 2; }
        if (!inSurf || !outSurf) { setErr(@"e8_attach: null surface"); return 3; }

        IOSurfaceRef ioIn = (IOSurfaceRef)inSurf, ioOut = (IOSurfaceRef)outSurf;
        CFRetain(ioIn); CFRetain(ioOut);
        id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(KIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(KIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(KReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @(proc));
        if (!req) { CFRelease(ioIn); CFRelease(ioOut); setErr(@"requestWithInputs -> nil"); return 4; }

        // 槽位编号 `proc * SLOTS_PER_PROC + slot`。8 个槽位不够：视觉塔按**瓦片**分槽，
        // s=256 时一张 1280² 的图有 25 个瓦片（preprocess.MAX_PIXELS 定的是 6 400 token）。
        // 64 个槽位 = 16 384 个 token，比引擎允许的任何图都多。
        NSNumber *key = @(proc * SLOTS_PER_PROC + slot);
        // replacing an existing binding: release the pair it held
        NSArray *old = p.sios[key];
        if (old) {
            CFRelease((CFTypeRef)[(NSValue *)old[0] pointerValue]);
            CFRelease((CFTypeRef)[(NSValue *)old[1] pointerValue]);
        }
        p.sreqs[key] = req;
        p.sios[key] = @[[NSValue valueWithPointer:ioIn], [NSValue valueWithPointer:ioOut]];
        if (slot == 0) {          // keep the flat accessors working
            p.reqs[@(proc)] = req;
            p.ios[@(proc)] = p.sios[key];
        }
        g_err[0] = 0;
    }
    return 0;
}

// A procedure may declare several `func_input`s.  The vision tower's `p1` takes the attention
// context and the residual as two separate faces, so neither has to be copied in first.  Two
// constraints, both measured (ticket 07 §7.19 / §7.23):
//   * the faces are bound to the declared inputs in **lexicographic order of the input names**,
//     not in declaration order — same rule as the procedure names;
//   * only procedure 0 gets its multi-input request wired.  Any later procedure that declares
//     more than one input fails with `Program Inference error 0x2` at eval time, so the tower
//     names its dual-input half `a_p1` and its single-input half `z_main`.
int e8_attach_n(const char *name, int proc, int slot, const uint64_t *ins,
                const int *indices, int count, uint64_t outSurf) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return 1;
        E8Prog *p = progGet(name);
        if (!p) { setErr(@"e8_attach_n: unknown program"); return 2; }
        if (!ins || !indices || count <= 0 || !outSurf) {
            setErr(@"e8_attach_n: null argument"); return 3;
        }
        if (slot < 0 || slot >= SLOTS_PER_PROC) {
            setErr(@"e8_attach_n: slot out of range"); return 3;
        }
        NSNumber *key = @(proc * SLOTS_PER_PROC + slot);
        NSMutableArray *w  = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
        NSMutableArray *ix = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
        for (int i = 0; i < count; i++) {
            IOSurfaceRef s = (IOSurfaceRef)ins[i];
            if (!s) { setErr(@"e8_attach_n: null surface"); return 3; }
            CFRetain(s);
            [w addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                          KIO, @selector(objectWithIOSurface:), s)];
            [ix addObject:@(indices[i])];
        }
        IOSurfaceRef ioOut = (IOSurfaceRef)outSurf;
        CFRetain(ioOut);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                      KIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(KReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            w, ix, @[wOut], @[@0], nil, nil, @(proc));
        if (!req) { setErr(@"e8_attach_n: requestWithInputs -> nil"); return 4; }

        // 记下这一槽位 retain 过的全部面（输入可能不止一张）：卸载时逐张 CFRelease。
        NSArray *old = p.sios[key];
        if (old) {
            for (NSValue *v in old) CFRelease((CFTypeRef)v.pointerValue);
        }
        NSMutableArray *held = [NSMutableArray arrayWithCapacity:(NSUInteger)count + 1];
        for (int i = 0; i < count; i++) {
            [held addObject:[NSValue valueWithPointer:(void *)(uintptr_t)ins[i]]];
        }
        [held addObject:[NSValue valueWithPointer:(void *)ioOut]];
        p.sreqs[key] = req;
        p.sios[key] = held;
        if (slot == 0) {          // keep the flat accessors working
            p.reqs[@(proc)] = req;
            p.ios[@(proc)] = held;
        }
        g_err[0] = 0;
    }
    return 0;
}

int e8_attach(const char *name, int proc, uint64_t inSurf, uint64_t outSurf) {
    return e8_attach_slot(name, proc, 0, inSurf, outSurf);
}

double e8_eval_slot(const char *name, int proc, int slot) {
    @autoreleasepool {
        E8Prog *p = progGet(name);
        if (!p) { setErr(@"e8_eval: unknown program"); return -1; }
        id req = p.sreqs[@(proc * SLOTS_PER_PROC + slot)];
        if (!req) { setErr(@"e8_eval_slot: (proc,slot) not attached"); return -1; }
        NSError *e = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            p.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double ms = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!ok) {
            setErr([NSString stringWithFormat:@"evaluate FAILED (proc %d slot %d): %@", proc, slot,
                    [e localizedDescription]]);
            return -1;
        }
        return ms;
    }
}

// Returns the ANE eval time in ms, or -1 on failure.
double e8_eval(const char *name, int proc) {
    @autoreleasepool {
        E8Prog *p = progGet(name);
        if (!p) { setErr(@"e8_eval: unknown program"); return -1; }
        id req = p.reqs[@(proc)];
        if (!req) { setErr(@"e8_eval: procedure not attached (call e8_attach first)"); return -1; }
        NSError *e = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            p.model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double ms = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!ok) {
            setErr([NSString stringWithFormat:@"evaluate FAILED (proc %d): %@", proc,
                    [e localizedDescription]]);
            return -1;
        }
        return ms;
    }
}

// Warm-up evals so the first timed call is not paying first-touch costs.
int e8_warmup(const char *name, int proc, int reps) {
    for (int i = 0; i < reps; i++) if (e8_eval(name, proc) < 0) return 1;
    return 0;
}

void e8_unload(const char *name) {
    @autoreleasepool {
        NSString *key = [NSString stringWithUTF8String:name];
        E8Prog *p = progGet(name);
        if (!p) return;
        NSError *e = nil;
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            p.model, @selector(unloadWithQoS:error:), 21, &e);
        // 每张被 retain 的面都要还回去：一张面一项输入 + 输出，而一个 procedure 可能有不止一张。
        for (NSArray *held in [p.sios allValues]) {
            for (NSValue *v in held) CFRelease((CFTypeRef)v.pointerValue);
        }
        [[NSFileManager defaultManager] removeItemAtPath:p.tmpDir error:nil];
        [g_lock lock];
        [g_progs removeObjectForKey:key];
        [g_lock unlock];
    }
}

// ---------------------------------------------------------------- can the content hash be injected?
//
// The descriptor keeps the identity as three separate hashes (`_networkTextHash`,
// `_weightsHash`, `_optionsPlistHash`) and `hexStringIdentifier` reports them back.  The 52 ms
// prefetch cost is those hashes being recomputed over 117.6 MB of weights on every load, for
// content that has not changed since the model was built -- i.e. work that could be done once,
// offline.  This probe answers whether the private API lets us supply them:
//
//   (1) normal path   : modelWithMILText:weights:optionsPlist:  -> hashes 117 MB, then read id
//   (2) injected path : alloc/init the descriptor and write the three hash ivars directly,
//                       keeping the same MIL/weights/options objects -> read id
//
// If (2) yields the same id in ~0 ms, the hash is offloadable; whether aned then accepts the
// model is answered by `compiledModelExists` + `loadWithQoS:` on it.

static void setIvarObject(id obj, const char *name, id val) {
    Ivar iv = class_getInstanceVariable(KDesc, name);
    if (iv) object_setIvar(obj, iv, val);
}

static id getIvarObject(id obj, const char *name) {
    Ivar iv = class_getInstanceVariable(KDesc, name);
    return iv ? object_getIvar(obj, iv) : nil;
}

int e8_hash_inject_probe(const char *milUtf8, const char *blobPath,
                         double *tNormalMs, double *tInjectMs,
                         char *idNormal, int idNormalLen,
                         char *idInject, int idInjectLen,
                         int *existsNormal, int *existsInject,
                         double *tLoadNormalMs, double *tLoadInjectMs,
                         double *tInitMs, double *tModelMs) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return 1;
        NSError *e = nil;
        NSData *milData = [[NSString stringWithUTF8String:milUtf8]
                              dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blobPath]
                                             options:NSDataReadingMappedIfSafe error:&e];
        if (!blob) { setErr(@"hash probe: blob read failed"); return 2; }
        NSDictionary *wdict = @{@"@model_path/weights.bin": @{@"offset": @0, @"data": blob}};

        // ---- (1) production path, then compile so aned holds this lowering
        uint64_t t0 = mach_absolute_time();
        id d1 = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            KDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!d1) { setErr(@"hash probe: descriptor nil"); return 3; }
        NSString *id1 = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(hexStringIdentifier));
        *tNormalMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        snprintf(idNormal, idNormalLen, "%s", id1.UTF8String ?: "");

        id ntH = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(networkTextHash));
        id wtH = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(weightsHash));
        id opH = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(optionsPlistHash));

        // ---- diagnostic: is the id derived from the three hash ivars, or cached on the object?
        NSString *ntH2 = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(networkTextHash));
        NSString *wtH2 = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(weightsHash));
        NSString *opH2 = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(optionsPlistHash));
        fprintf(stderr, "[probe] d1 hashes: nt=%s wt=%s op=%s\n",
                ntH2 ? ntH2.UTF8String : "(nil)", wtH2 ? wtH2.UTF8String : "(nil)",
                opH2 ? opH2.UTF8String : "(nil)");
        setIvarObject(d1, "_weightsHash", @"deadbeef");
        NSString *idMut = ((id(*)(id,SEL))objc_msgSend)(d1, @selector(hexStringIdentifier));
        setIvarObject(d1, "_weightsHash", wtH2);
        fprintf(stderr, "[probe] id changes when _weightsHash is replaced? %s\n",
                [idMut isEqualToString:id1] ? "NO (cached)" : "YES (derived)");

        id m1 = ((id(*)(Class,SEL,id))objc_msgSend)(KIMM, @selector(inMemoryModelWithDescriptor:), d1);
        if (!m1) { setErr(@"hash probe: model nil"); return 4; }
        SEL exSel = sel_getUid("compiledModelExists");
        *existsNormal = ((BOOL(*)(id,SEL))objc_msgSend)(m1, exSel) ? 1 : 0;
        if (!*existsNormal) {
            NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:id1];
            NSFileManager *fm = [NSFileManager defaultManager];
            [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
          withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
            [blob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights.bin"] atomically:YES];
            BOOL cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                m1, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            if (!cok) { setErr([NSString stringWithFormat:@"probe compile failed: %@", e8ErrText(e)]); return 5; }
        }

        // ---- (2) descriptor assembled by hand: alloc/init + ivars, no hashing at all
        t0 = mach_absolute_time();
        // `[[KDesc alloc] init]` returns nil (the class only supports its designated inits), so
        // the instance is created uninitialised and every field we need is written by hand.
        id d2 = [KDesc alloc];          // alloc, but never init: init hashes, and it returns nil
        fprintf(stderr, "[probe] d2=%p\n", d2);
        setIvarObject(d2, "_networkText", milData);
        setIvarObject(d2, "_weights", wdict);
        setIvarObject(d2, "_optionsPlist", nil);
        setIvarObject(d2, "_networkTextHash", ntH);
        setIvarObject(d2, "_weightsHash", wtH);
        setIvarObject(d2, "_optionsPlistHash", opH);
        @try {
            [d2 setValue:@YES forKey:@"isMILModel"];
        } @catch (NSException *ex) {
            fprintf(stderr, "[probe] KVC isMILModel failed: %s\n", ex.reason.UTF8String);
        }
        *tInitMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        fprintf(stderr, "[probe] d2 assembled in %.2f ms\n", *tInitMs);
        fprintf(stderr, "[probe] reading id of d2...\n");
        NSString *id2 = ((id(*)(id,SEL))objc_msgSend)(d2, @selector(hexStringIdentifier));
        fprintf(stderr, "[probe] d2 id=%s\n", id2.UTF8String ?: "(nil)");
        *tInjectMs = 0;
        snprintf(idInject, idInjectLen, "%s", id2.UTF8String ?: "");

        t0 = mach_absolute_time();
        id m2 = ((id(*)(Class,SEL,id))objc_msgSend)(KIMM, @selector(inMemoryModelWithDescriptor:), d2);
        *tModelMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!m2) { setErr(@"hash probe: injected model nil"); return 6; }
        *existsInject = ((BOOL(*)(id,SEL))objc_msgSend)(m2, exSel) ? 1 : 0;

        // ---- (3) both loads, from the same warm cache
        t0 = mach_absolute_time();
        BOOL ok1 = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            m1, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        *tLoadNormalMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (ok1) ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            m1, @selector(unloadWithQoS:error:), 21, &e);

        t0 = mach_absolute_time();
        BOOL ok2 = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            m2, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        *tLoadInjectMs = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (ok2) ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            m2, @selector(unloadWithQoS:error:), 21, &e);
        if (!ok1 || !ok2) {
            setErr([NSString stringWithFormat:@"hash probe: load failed %d/%d (%@)", ok1, ok2,
                    [e localizedDescription]]);
            return 7;
        }
        g_err[0] = 0;
    }
    return 0;
}

// ---------------------------------------------------------------- offline content hashes
//
// The descriptor's identity is just three SHA-256 strings (`networkTextHash`, `weightsHash`,
// `optionsPlistHash`) joined by '_', and `initWithNetworkText:...` computes them -- 44-51 ms of
// hashing over 117.6 MB of weights -- on every single load, for content that has not changed
// since the model was built.  Measured (../e3-attention-results.md §10.12):
//
//   * assembling a descriptor by hand (alloc, never init, then writing the ivars) costs 0.26 ms
//   * with the three strings supplied, the id comes out bit-identical to the normal path
//   * aned accepts it: `compiledModelExists` = 1 and `loadWithQoS:` succeeds in 4.3 ms
//
// so the hash is a *build-time* quantity, not a load-time one.  `e8_descriptor_hashes` produces
// the three strings once (paying the 50 ms); `e8_load_fast` then loads with them.  Both are
// strictly optimisations: if aned does not hold the lowering, e8_load_fast fails with rc 7 and
// the caller falls back to `e8_load_warm`, which is what the engine does today.

int e8_descriptor_hashes(const char *milUtf8, const char *blobPath,
                         char *nt, int ntLen, char *wt, int wtLen, char *op, int opLen,
                         double *ms) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return 1;
        NSError *e = nil;
        NSData *milData = [[NSString stringWithUTF8String:milUtf8]
                              dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blobPath]
                                             options:NSDataReadingMappedIfSafe error:&e];
        if (!blob) { setErr(@"descriptor_hashes: blob read failed"); return 2; }
        NSDictionary *wdict = @{@"@model_path/weights.bin": @{@"offset": @0, @"data": blob}};
        uint64_t t0 = mach_absolute_time();
        id d = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            KDesc, @selector(modelWithMILText:weights:optionsPlist:), milData, wdict, nil);
        if (!d) { setErr(@"descriptor_hashes: descriptor nil"); return 3; }
        NSString *a = ((id(*)(id,SEL))objc_msgSend)(d, @selector(networkTextHash));
        NSString *b = ((id(*)(id,SEL))objc_msgSend)(d, @selector(weightsHash));
        NSString *c = ((id(*)(id,SEL))objc_msgSend)(d, @selector(optionsPlistHash));
        if (ms) *ms = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        snprintf(nt, ntLen, "%s", a.UTF8String ?: "");
        snprintf(wt, wtLen, "%s", b.UTF8String ?: "");
        snprintf(op, opLen, "%s", c.UTF8String ?: "");
        g_err[0] = 0;
    }
    return 0;
}

// Load by (MIL, weights, three known hashes): no hashing, no compileWithQoS.
// rc: 0 ok | 1 no init | 2 bad arg | 3 already loaded | 4 blob | 5 descriptor | 6 model
//     7 aned has no lowering for this identity (caller should fall back)
int e8_load_fast(const char *name, const char *milUtf8, const char *blobPath,
                 const char *ntHash, const char *wtHash, const char *opHash,
                 double *loadMsOut) {
    @autoreleasepool {
        if (!KDesc && e8_init() != 0) return 1;
        if (!name || !milUtf8 || !blobPath || !ntHash || !wtHash || !opHash) {
            setErr(@"load_fast: null argument"); return 2;
        }
        NSString *key = [NSString stringWithUTF8String:name];
        [g_lock lock];
        BOOL dup = g_progs[key] != nil;
        [g_lock unlock];
        if (dup) { setErr(@"load_fast: already loaded"); return 3; }

        NSError *e = nil;
        NSData *milData = [[NSString stringWithUTF8String:milUtf8]
                              dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blob = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:blobPath]
                                             options:NSDataReadingMappedIfSafe error:&e];
        if (!blob) { setErr(@"load_fast: blob read failed"); return 4; }
        NSDictionary *wdict = @{@"@model_path/weights.bin": @{@"offset": @0, @"data": blob}};

        id desc = [KDesc alloc];       // alloc without init: init hashes, and it returns nil
        setIvarObject(desc, "_networkText", milData);
        setIvarObject(desc, "_weights", wdict);
        setIvarObject(desc, "_optionsPlist", nil);
        setIvarObject(desc, "_networkTextHash", [NSString stringWithUTF8String:ntHash]);
        setIvarObject(desc, "_weightsHash", [NSString stringWithUTF8String:wtHash]);
        setIvarObject(desc, "_optionsPlistHash", [NSString stringWithUTF8String:opHash]);
        @try {
            [desc setValue:@YES forKey:@"isMILModel"];
        } @catch (NSException *ex) {
            setErr([NSString stringWithFormat:@"load_fast: KVC failed: %@", ex.reason]);
            return 5;
        }
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(KIMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { setErr(@"load_fast: inMemoryModelWithDescriptor -> nil"); return 6; }

        SEL exSel = sel_getUid("compiledModelExists");
        if (![model respondsToSelector:exSel] ||
            !((BOOL(*)(id,SEL))objc_msgSend)(model, exSel)) {
            setErr(@"load_fast: aned has no lowering for this identity (fall back)");
            return 7;
        }
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double ms = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!ok) {
            setErr([NSString stringWithFormat:@"load_fast: loadWithQoS FAILED: %@", e8ErrText(e)]);
            return 8;
        }
        E8Prog *p = [E8Prog new];
        p.model = model;
        p.tmpDir = nil;                 // nothing was written to disk on this path
        p.compileMs = 0;
        p.loadMs = ms;
        p.cacheHit = 1;
        p.reqs = [NSMutableDictionary dictionary];
        p.ios = [NSMutableDictionary dictionary];
        p.sreqs = [NSMutableDictionary dictionary];
        p.sios = [NSMutableDictionary dictionary];
        [g_lock lock];
        g_progs[key] = p;
        [g_lock unlock];
        if (loadMsOut) *loadMsOut = ms;
        g_err[0] = 0;
    }
    return 0;
}

// ---------------------------------------------------------------- suspend / resume
//
// What a prefetch actually costs today (../e3-attention-results.md §10.10) is 57 ms/layer, of
// which only 4.5 ms is `loadWithQoS:`; the other 52 ms is rebuilding the descriptor from the
// MIL text + weights and computing the content hash (`hexStringIdentifier`) -- the exact same
// object, 20 times per pass.  If `unloadWithQoS:` only drops the device-side residency and
// leaves the model object usable, that 52 ms is pure repetitive work we can cache away:
//
//   e8_suspend  unloadWithQoS: only -- keep the model, the descriptors and the request bindings
//   e8_resume   loadWithQoS: only  -- no descriptor rebuild, no content hash
//
// `e8_unload` stays the full teardown (release surfaces, delete tmpDir, drop the object).
// Memory question this has to answer: does suspend give the wired memory back the way unload
// does (§10.9)?  If not, keeping models alive is not free and the idea is dead.

int e8_suspend(const char *name) {
    @autoreleasepool {
        E8Prog *p = progGet(name);
        if (!p) { setErr(@"e8_suspend: unknown program"); return 1; }
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            p.model, @selector(unloadWithQoS:error:), 21, &e);
        if (!ok) {
            setErr([NSString stringWithFormat:@"suspend(unloadWithQoS) FAILED: %@", e8ErrText(e)]);
            return 2;
        }
        p.suspended = 1;
        g_err[0] = 0;
    }
    return 0;
}

// Resume a suspended model: bare loadWithQoS:, timed into *loadMs.
int e8_resume(const char *name, double *loadMs) {
    @autoreleasepool {
        E8Prog *p = progGet(name);
        if (!p) { setErr(@"e8_resume: unknown program"); return 1; }
        if (!p.suspended) { setErr(@"e8_resume: program is not suspended"); return 3; }
        NSError *e = nil;
        uint64_t t0 = mach_absolute_time();
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            p.model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        double ms = (double)(mach_absolute_time() - t0) * g_nsPerTick / 1e6;
        if (!ok) {
            setErr([NSString stringWithFormat:@"resume(loadWithQoS) FAILED: %@", e8ErrText(e)]);
            return 4;
        }
        p.suspended = 0;
        p.resumes++;
        p.resumeMs = ms;
        if (loadMs) *loadMs = ms;
        g_err[0] = 0;
    }
    return 0;
}

// Full-time statistics for the suspend/resume loop: how many times, and the last resume cost.
int e8_suspend_stats(const char *name, int *suspended, int *resumes, double *lastResumeMs) {
    E8Prog *p = progGet(name);
    if (!p) { setErr(@"e8_suspend_stats: unknown program"); return 1; }
    if (suspended) *suspended = p.suspended;
    if (resumes) *resumes = p.resumes;
    if (lastResumeMs) *lastResumeMs = p.resumeMs;
    return 0;
}

void e8_unload_all(void) {
    @autoreleasepool {
        [g_lock lock];
        NSArray *keys = [g_progs allKeys];
        [g_lock unlock];
        for (NSString *k in keys) e8_unload(k.UTF8String);
    }
}

int e8_count(void) {
    [g_lock lock];
    int n = (int)g_progs.count;
    [g_lock unlock];
    return n;
}

// ---------------------------------------------------------------- memory observability

// free + inactive + speculative + purgeable: what the kernel can still hand out
double e8_avail_gb(void) {
    mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t v;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &c) != KERN_SUCCESS) return -1;
    return ((double)v.free_count + v.inactive_count + v.speculative_count + v.purgeable_count)
           * 16384.0 / 1e9;
}

// wired = non-pageable; this is what exhausted the machine in the E1c panics
double e8_wired_gb(void) {
    mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t v;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &c) != KERN_SUCCESS) return -1;
    return (double)v.wire_count * 16384.0 / 1e9;
}

// resident footprint of THIS process (what a user would see in Activity Monitor)
double e8_rss_gb(void) {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &c) != KERN_SUCCESS) return -1;
    return (double)info.resident_size / 1e9;
}

double e8_peak_rss_gb(void) {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t c = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &c) != KERN_SUCCESS) return -1;
    return (double)info.resident_size_max / 1e9;
}

// phys_footprint = what Activity Monitor calls "Memory": dirty + compressed + IOKit
// (incl. our IOSurfaces). This is the number the 4 GB budget is about.
double e8_footprint_gb(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &c) != KERN_SUCCESS) return -1;
    return (double)info.phys_footprint / 1e9;
}

double e8_peak_footprint_gb(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t c = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &c) != KERN_SUCCESS) return -1;
    return (double)info.ledger_phys_footprint_peak / 1e9;
}

// high-water mark of the process footprint, sampled by the caller each iteration instead —
// ledger_phys_footprint_peak is not available on every macOS build, so expose the sampler.
int e8_peak_reset(void) { return 0; }
