# simplewebp

Vendored from [MikuAuahDark/simplewebp](https://github.com/MikuAuahDark/simplewebp)
at commit `d1a728a1f8ec7348ca2a5039b6dd813b83986fbb` (header version `20260718`).
The BSD-3-Clause license and WebM patent grant are embedded in `simplewebp.h`.

Local changes bound memory and chunk reads, validate declared chunk sizes before
allocation, reject incomplete trailing chunks and missing padding, account for
the ten-byte VP8 frame header when validating partition lengths, and release
chunk handles on allocation errors or duplicate image/alpha chunks.

Regression coverage lives in `tests/unit/image_inputs_test.cpp` and can be run
under AddressSanitizer/UndefinedBehaviorSanitizer. Keep these fixes when updating
the vendored header.
