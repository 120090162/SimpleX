# The Simple-X: a Simulation based on [Simple](https://github.com/Simple-Robotics/Simple)

# Environment Setup
```bash
conda create -n simplex python=3.10 -c conda-forge -y
conda activate simplex
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake make git pkg-config -y
conda install -c conda-forge eigen boost urdfdom hpp-fcl console_bridge -y # auto install coal
conda install -c conda-forge eigenpy numpy -y

git clone --recursive https://github.com/120090162/SimpleX.git
# 安装特定版本的pinocchio
cd SimpleX/third_party/pinocchio

mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_COLLISION_SUPPORT=ON \
    -DBUILD_PYTHON_INTERFACE=ON \
    -DPYTHON_EXECUTABLE=$(which python)

make -j4 2>&1 | tee build.log
make install

# 安装Clarabel库
cd ../../Clarabel

conda install -c conda-forge rust -y
cargo --version # show version >= 1.92.0
# add 'export PATH=$PATH:$HOME/.cargo/bin' to ~/.bashrc or ~/.zshrc
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release
cmake --build .
make install

# 安装simplex库
cd ../../../

conda install -c conda-forge glfw glew mesalib -y
conda install -c conda-forge spdlog fmt -y
conda install -c conda-forge pybind11 -y
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DPYTHON_EXECUTABLE=$(which python)

make -j4 2>&1 | tee build.log
make install

# [optional] boost test
conda install -c conda-forge fmt -y
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DBUILD_TEST_CASES=ON \
    -DPYTHON_EXECUTABLE=$(which python)

make -j4 2>&1 | tee build.log
ctest # ctest -V

# [optional] benchmark
conda install -c conda-forge google-benchmark -y
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -BUILD_BENCHMARKS=ON \
    -DPYTHON_EXECUTABLE=$(which python)
make -j4 2>&1 | tee build.log
./benchmarks/affine-transform
```

# 测试例子

```bash
conda install -c conda-forge mujoco matplotlib -y
conda install -c conda-forge typed-argument-parser robot_descriptions meshcat-python -y

meshcat-server
```