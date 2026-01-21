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
make install # update the libsimplex.so file
ctest # ctest -V

# [optional] benchmark
conda install -c conda-forge benchmark -y
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -BUILD_BENCHMARKS=ON \
    -DPYTHON_EXECUTABLE=$(which python)
make -j4 2>&1 | tee build.log
make install # update the libsimplex.so file
./benchmarks/affine-transform

# python binding
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TEST_CASES=ON \
    -DPYTHON_EXECUTABLE=$(which python)
make -j4 2>&1 | tee build.log
make install # update the libsimplex.so file
cd ..
pip install -e . # install simplex library
```

# 测试例子

```bash
conda install -c conda-forge mujoco matplotlib -y
conda install -c conda-forge typed-argument-parser robot_descriptions meshcat-python -y
pip install "imageio[ffmpeg]"==2.37.2 imageio-ffmpeg==0.6.0

meshcat-server
# another terminal
python simplex_sandbox/forward/cartpole.py
```

# 测试cimpc案例
```bash
# 安装特定crocoddyl
git clone -b m3.2.0 --recursive https://github.com/120090162/crocoddyl.git third_party/crocoddyl
cd third_party/crocoddyl
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_BENCHMARK=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_WITH_IPOPT=OFF \
    -DGENERATE_PYTHON_STUBS=ON \
    -DBUILD_PYTHON_INTERFACE=ON \
    -DBUILD_WITH_MULTITHREADS=ON \
    -DPYTHON_EXECUTABLE=$(which python)
make -j4
make install
# 测试crocoddyl安装是否正确
cd ..
python examples/double_pendulum_fwddyn.py plot

# 测试flip demo

# 测试walk demo

# 测试tolerant demo
```

# TODO
## short term plan
- [ ] 加入diffcoal的支持
- [x] 加入pybind
- [x] 加入四足demo
- [ ] 加入柔性梯度
- [ ] 加入cimpc crocoddyl实现
- [ ] 完成四足行走测试
- [ ] 加入mujoco sim to sim 测试
## middle term plan
- [ ] 完成go2 isaacgym训练
- [ ] 完成go2 状态判别器训练
- [ ] 完成go2 cost 设计
## final term plan
- [ ] 完成容错框架
- [ ] 完成flow matching policy的训练
- [ ] 完成go2 实物部署