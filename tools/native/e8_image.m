// e8_image.m -- decode PNG/JPEG/HEIC/TIFF bytes to an 8-bit RGB buffer, for the vision path's S1.
//
// Why a C shim at all: the service's venv has no pillow, no torch and no OpenCV (checked), and
// pure-Python JPEG decoding is not a thing anyone should ship.  ImageIO is the system decoder --
// it covers every format the checkpoint's processor accepts, and it is the same decoder Preview
// and Finder use, so "the file opens on this Mac" and "the service can read it" are one question.
//
// Two decisions that are deliberate, because they must match the reference processor
// (`Qwen2VLImageProcessorPil`, which is what §4.22 measured our pixels against):
//
//  1. **EXIF orientation is NOT applied.**  `CGImageSourceCreateImageAtIndex` ignores the
//     orientation tag, and so does Pillow's `Image.open(...).convert("RGB")`.  The two libraries
//     therefore agree by default.  Applying the transform here would make our pixels differ from
//     the reference on exactly the files where it matters most (phone photographs), so if that is
//     ever wanted it has to change in both places at once.
//  2. **Alpha is dropped, not composited.**  `kCGImageAlphaNoneSkipLast` gives a destination with
//     no alpha, so a source's alpha is ignored rather than blended over a background -- the same
//     thing Pillow's `convert("RGB")` does.  Compositing over white would be a different function.
//
// The output is RGBA8 in sRGB with the alpha byte unused; Python drops the fourth byte.  Keeping
// 4 bytes here costs a third more memory and saves a per-format conversion branch in C.
//
//   clang -dynamiclib -fobjc-arc -O2 -framework Foundation -framework ImageIO \
//         -framework CoreGraphics -o /tmp/e8/libe8image.dylib e8_image.m
//
// `e8_img_decode` returns 0 on success; on failure it writes a message into `err` and returns
// non-zero, which Python turns into a ValueError.  The caller owns `*out` and must release it with
// `e8_img_free` -- the buffer is malloc'd here, so Python must not free it with libc's free even
// though on this platform that would happen to work.

#include <stdlib.h>
#include <string.h>
#include <ImageIO/ImageIO.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>

static void set_err(char *err, size_t errlen, const char *msg) {
    if (err && errlen) {
        strncpy(err, msg, errlen - 1);
        err[errlen - 1] = '\0';
    }
}

int e8_img_decode(const uint8_t *data, size_t len,
                  uint8_t **out, int *out_w, int *out_h,
                  char *err, size_t errlen) {
    if (out) *out = NULL;
    if (!data || len == 0) { set_err(err, errlen, "empty image data"); return 1; }

    // No copy: the CFData borrows Python's bytes, which outlive the call.
    CFDataRef cf = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, data, (CFIndex)len,
                                              kCFAllocatorNull);
    if (!cf) { set_err(err, errlen, "cannot wrap the image bytes"); return 2; }

    CGImageSourceRef src = CGImageSourceCreateWithData(cf, NULL);
    if (!src) { CFRelease(cf); set_err(err, errlen, "not an image (no decoder claimed it)"); return 3; }

    if (CGImageSourceGetCount(src) < 1) {
        CFRelease(src); CFRelease(cf);
        set_err(err, errlen, "the image source holds no frames");
        return 4;
    }

    // Index 0 is the primary image.  A multi-frame file (animated GIF, HEIC burst) is read as its
    // first frame: the processor takes one image per placeholder, and picking a "best" frame would
    // be a policy the reference does not have.
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
    if (!img) {
        CFRelease(src); CFRelease(cf);
        set_err(err, errlen, "the frame could not be decoded");
        return 5;
    }

    size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    if (w == 0 || h == 0 || w > 100000 || h > 100000) {
        CGImageRelease(img); CFRelease(src); CFRelease(cf);
        set_err(err, errlen, "the image has no usable dimensions");
        return 6;
    }

    uint8_t *buf = (uint8_t *)malloc(w * h * 4);
    if (!buf) {
        CGImageRelease(img); CFRelease(src); CFRelease(cf);
        set_err(err, errlen, "out of memory for the decoded image");
        return 7;
    }

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    // AlphaNoneSkipLast: the destination has no alpha, so the source's is discarded (see the note
    // at the top).  Byte order default keeps the channels in R,G,B order for 8-bit components.
    CGContextRef ctx = CGBitmapContextCreate(buf, w, h, 8, w * 4, cs,
                                             kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);
    if (!ctx) {
        free(buf);
        CGColorSpaceRelease(cs); CGImageRelease(img); CFRelease(src); CFRelease(cf);
        set_err(err, errlen, "cannot create the RGB drawing context");
        return 8;
    }

    CGContextDrawImage(ctx, CGRectMake(0, 0, (CGFloat)w, (CGFloat)h), img);

    CGContextRelease(ctx);
    CGColorSpaceRelease(cs);
    CGImageRelease(img);
    CFRelease(src);
    CFRelease(cf);

    if (out) *out = buf; else free(buf);
    if (out_w) *out_w = (int)w;
    if (out_h) *out_h = (int)h;
    return 0;
}

void e8_img_free(void *p) { free(p); }

// e8_img_size -- the pixel dimensions of an encoded image, without decoding it.
//
// Why this exists: the front door refuses images over 16 MP (`MAX_IMAGE_SOURCE_PIXELS`), and
// asking `e8_img_decode` first would allocate exactly the buffer that limit is there to bound --
// a 30 000 x 30 000 PNG compresses to a few hundred kilobytes and 3.6 GB once decoded.  ImageIO
// answers from the container's header, which is what Pillow's ``Image.open`` does for the same
// check upstream.  Same conventions as `e8_img_decode`: index 0 of a multi-frame file, and the
// encoded (not EXIF-rotated) size, which is also what ``Image.size`` reports.
int e8_img_size(const uint8_t *data, size_t len,
                int *out_w, int *out_h,
                char *err, size_t errlen) {
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    if (!data || len == 0) { set_err(err, errlen, "empty image data"); return 1; }

    CFDataRef cf = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, data, (CFIndex)len,
                                              kCFAllocatorNull);
    if (!cf) { set_err(err, errlen, "cannot wrap the image bytes"); return 2; }

    CGImageSourceRef src = CGImageSourceCreateWithData(cf, NULL);
    if (!src) { CFRelease(cf); set_err(err, errlen, "not an image (no decoder claimed it)"); return 3; }

    int w = 0, h = 0;
    CFDictionaryRef props = CGImageSourceCopyPropertiesAtIndex(src, 0, NULL);
    if (props) {
        CFNumberRef pw = (CFNumberRef)CFDictionaryGetValue(props, kCGImagePropertyPixelWidth);
        CFNumberRef ph = (CFNumberRef)CFDictionaryGetValue(props, kCGImagePropertyPixelHeight);
        if (pw) CFNumberGetValue(pw, kCFNumberIntType, &w);
        if (ph) CFNumberGetValue(ph, kCFNumberIntType, &h);
        CFRelease(props);
    }
    CFRelease(src);
    CFRelease(cf);

    if (w <= 0 || h <= 0) {
        set_err(err, errlen, "the image reports no pixel dimensions");
        return 6;
    }
    if (out_w) *out_w = w;
    if (out_h) *out_h = h;
    return 0;
}
