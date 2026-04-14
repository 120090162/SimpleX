# The Simple-X: a Simulation based on [Simple](https://github.com/Simple-Robotics/Simple)

# Environment Setup
```bash
conda create -n simplex python=3.10 -c conda-forge -y
conda activate simplex
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake make git pkg-config -y
conda install -c conda-forge eigen=3.4.0 eigenpy boost urdfdom hpp-fcl console_bridge -y # auto install coal
conda install -c conda-forge numpy -y

git clone --recursive https://github.com/120090162/SimpleX.git
# 安装特定版本的pinocchio
cd SimpleX/third_party/pinocchio

mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_COLLISION_SUPPORT=ON \
    -DBUILD_WITH_SDF_SUPPORT=OFF \
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
    -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_PYTHON_INTERFACE=OFF \
    -DBUILD_BENCHMARKS=ON \
    -DPYTHON_EXECUTABLE=$(which python)
make -j4 2>&1 | tee build.log
make install # update the libsimplex.so file
./benchmarks/simplex-benchmark-affine-transform
./benchmarks/simplex-benchmark-mujoco-humanoid

# python binding
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TEST_CASES=OFF \
    -DBUILD_PYTHON_INTERFACE=ON \
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
# forward test
python simplex_sandbox/forward/cartpole.py
# derivatives test
python simplex_sandbox/derivatives/go2_contact_id.py
```

# 测试cimpc案例
```bash
# 安装contactbench, ref to https://github.com/120090162/a1-cimpc-control/tree/dev
conda install -c conda-forge line_profiler proxsuite simde cereal -y
git clone -b dev --recursive https://github.com/120090162/ContactBench.git third_party/ContactBench
cd third_party/ContactBench
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python)
make -j4
make install
# 测试contactbench安装是否正确
cd ../../..
conda install -c conda-forge loop-rate-limiters cvxpy -y # make sure cvxpy >= 1.7.2
# display xml model
python examples/display_model.py --xml_path=unitree_go2/scene_display.xml
# test contact force
python examples/test_force.py
# test_cb_simulate
python examples/test_simulate.py --record --save --plot
# test cb_solvers
python examples/test_solvers.py

# 注意:
# 会有小概率出现报 Fatal Python error: Segmentation fault
# python 数值精度问题, 多式几次就好


# 安装特定crocoddyl, make sure the pos is under Simplex
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

# go2 sim demo
python demo/go2_sim.py
python demo/go2_sim.py --robot_path="unitree_go2/mjcf/go2.xml"
```

# 测试指令

cimpc_sandbox下存放着旧cimpc的代码

```bash
cd cimpc_sandbox
mkdir build && cd build
cmake ..
make -j4
ctest
```

`calcDiff_test.ipynb`与`cimpc_test.ipynb`可以检测cimpc算法使用是否合理

## 小工具
```bash
# meshcat端口清除脚本
chmod +x scripts/kill_meshcat_ports.sh
./scripts/kill_meshcat_ports.sh
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