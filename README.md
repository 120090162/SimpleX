# The Simple-X: a Simulation based on [Simple](https://github.com/Simple-Robotics/Simple)

# Environment Setup
```bash
conda create -n simplex python=3.10 -c conda-forge -y
conda activate simplex
conda install -c conda-forge compilers cmake make git pkg-config -y
conda install -c conda-forge eigen boost urdfdom hpp-fcl console_bridge -y
conda install -c conda-forge eigenpy numpy -y

# 安装特定版本的pinocchio
git clone --recursive https://github.com/120090162/SimpleX.git
cd SimpleX/third_party/pinocchio

mkdir build && cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_COLLISION_SUPPORT=ON \
    -DBUILD_PYTHON_INTERFACE=ON \
    -DPYTHON_EXECUTABLE=$(which python)

make -j4
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

make -j4
make install
```

# 测试例子

```bash
conda install -c conda-forge mujoco matplotlib -y
conda install -c conda-forge typed-argument-parser robot_descriptions meshcat-python -y

meshcat-server
```