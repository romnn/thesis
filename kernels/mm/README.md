#### CUDA Matrix multiplication

This is an inefficient, native, and trivial implmentation of a matrix multiplication in CUDA.

The goal is to emulate this CUDA kernel using a modern rust based emulator.

```bash
make -j
# should show RPATH as LAZY, not NOW
scanelf -a _build/mm

# check if cuda runtime was dynamically linked
ldd _build/mm
# should show libcudart.so.10.1 => /lib/x86_64-linux-gnu/libcudart.so.10.1 or similar

# this will only work if the shared cuda runtime library is used
ltrace -e "*cuda*" _build/mm

# get a copy of all the shared library calls
ltrace -e "*cuda*" _build/mm 2> libcudart_calls.txt
```
