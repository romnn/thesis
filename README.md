#### Roadmap
- create simple CUDA program (matrix multiply) written in C++
- use `strace` or so to find the function calls into `libcudart.so`
- build a rust shared lib `libcudart.so` as a crash test dummy
- link simple example against dummy library and see if it works

```bash
sudo apt-get upgrade ninja-build
pip3 install --upgrade --user meson
```

Run the tests in a pre-configured CUDA environment
```bash
docker build --push -t gpucache .
docker run --rm -i -v "$PWD":/src gpucache cargo test
```

#### Compile GPGPU-Sim
```
sudo rm -rf /usr/local/go/
tar -C /usr/local -xzf go1.11.4.linux-amd64.tar.gz
go install github.com/bazelbuild/bazelisk@latest
ln -s /home/roman/go/bin/bazelisk /home/roman/.local/bin/bazel
```
```bash
export CUDA_INSTALL_PATH=/usr/lib/cuda
sudo cp /usr/bin/nvcc /usr/lib/cuda/bin/nvcc

# dependencies for ubuntu
sudo apt-get install build-essential xutils-dev bison zlib1g-dev flex libglu1-mesa-dev
# CUDA SDK dependencies
sudo apt-get install libxi-dev libxmu-dev libglut3-dev

make -j

# this will compile a shared library for CUDA and set LD_LIBRARY_PATH to there
/lib/gcc-8.4.0/cuda-10010/debug/libcudart.so
```
