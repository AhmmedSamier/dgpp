# dr_flac

DGPP vendors `third_party/dr_flac/dr_flac.h` from
[mackron/dr_libs, commit dfe8377631000664666519fdb83da193fd8037f4](https://github.com/mackron/dr_libs/blob/dfe8377631000664666519fdb83da193fd8037f4/dr_flac.h).
Only the memory decoder with float output is compiled
(`DR_FLAC_IMPLEMENTATION`, `drflac_open_memory` /
`drflac_read_pcm_frames_f32`). The upstream file is choice-of-license
public domain or MIT No Attribution; DGPP distributes it under MIT No
Attribution:

Copyright 2023 David Reid

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
