#include <benchmark/benchmark.h>
#include <Eigen/Core>

#define N 100
#define WARMUP_TIME 1.
#define MIN_TIME 2.

using Vec3f = Eigen::Matrix<float, 3, 1>;
using Mat3f = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using Vec4f = Eigen::Matrix<float, 4, 1>;
using Mat4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using Mat34f = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;

using Vec3d = Eigen::Matrix<double, 3, 1>;
using Mat3d = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
using Vec4d = Eigen::Matrix<double, 4, 1>;
using Mat4d = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>;
using Mat34d = Eigen::Matrix<double, 3, 4, Eigen::RowMajor>;

using Vec3i16 = Eigen::Matrix<std::int16_t, 3, 1>;
using Mat3i16 = Eigen::Matrix<std::int16_t, 3, 3, Eigen::RowMajor>;
using Vec4i16 = Eigen::Matrix<std::int16_t, 4, 1>;
using Mat4i16 = Eigen::Matrix<std::int16_t, 4, 4, Eigen::RowMajor>;
using Mat34i16 = Eigen::Matrix<std::int16_t, 3, 4, Eigen::RowMajor>;

template<typename ResVec, typename InVec, typename Mat>
struct BenchFixture : benchmark::Fixture
{
  void SetUp(benchmark::State &)
  {
    y.reserve(N);
    x.reserve(N);
    t.reserve(N);
    M.reserve(N);
    dotprod.reserve(N);
    for (std::size_t i = 0; i < N; ++i)
    {
      y.push_back(ResVec::Zero());
      x.push_back(InVec::Random());
      t.push_back(InVec::Random());
      M.push_back(Mat::Random());
      dotprod.push_back(0);
    }
  }

  void TearDown(benchmark::State &)
  {
  }

  // y = sum_i M_i * x_i + t_i
  std::vector<ResVec, Eigen::aligned_allocator<ResVec>> y;
  std::vector<InVec, Eigen::aligned_allocator<InVec>> x;
  std::vector<InVec, Eigen::aligned_allocator<InVec>> t;
  std::vector<Mat, Eigen::aligned_allocator<Mat>> M;
  std::vector<typename ResVec::Scalar> dotprod;
};

using Bench33f = BenchFixture<Vec3f, Vec3f, Mat3f>;
BENCHMARK_DEFINE_F(Bench33f, Mat3Vec3f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].noalias() = M[i] * x[i] + t[i];
    }
  }
}
BENCHMARK_REGISTER_F(Bench33f, Mat3Vec3f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using Bench33d = BenchFixture<Vec3d, Vec3d, Mat3d>;
BENCHMARK_DEFINE_F(Bench33d, Mat3Vec3d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].noalias() = M[i] * x[i] + t[i];
    }
  }
}
BENCHMARK_REGISTER_F(Bench33d, Mat3Vec3d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using Bench44f = BenchFixture<Vec4f, Vec4f, Mat4f>;
BENCHMARK_DEFINE_F(Bench44f, Mat4Vec4f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].noalias() = M[i] * x[i];
    }
  }
}
BENCHMARK_REGISTER_F(Bench44f, Mat4Vec4f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using Bench44d = BenchFixture<Vec4d, Vec4d, Mat4d>;
BENCHMARK_DEFINE_F(Bench44d, Mat4Vec4d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].noalias() = M[i] * x[i];
    }
  }
}
BENCHMARK_REGISTER_F(Bench44d, Mat4Vec4d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using Bench34f = BenchFixture<Vec3f, Vec4f, Mat34f>;
BENCHMARK_DEFINE_F(Bench34f, Mat34Vec3f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].noalias() = M[i] * x[i];
    }
  }
}
BENCHMARK_REGISTER_F(Bench34f, Mat34Vec3f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using Bench34d = BenchFixture<Vec3d, Vec4d, Mat34d>;
BENCHMARK_DEFINE_F(Bench34d, Mat34Vec3d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].noalias() = M[i] * x[i];
    }
  }
}
BENCHMARK_REGISTER_F(Bench34d, Mat34Vec3d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using YannBench34f = BenchFixture<Vec4f, Vec4f, Mat34f>;
BENCHMARK_DEFINE_F(YannBench34f, Mat34Vec3f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].head<3>().noalias() = M[i] * x[i];
      y[i].coeffRef(3) = 1.0f;
    }
  }
}
BENCHMARK_REGISTER_F(YannBench34f, Mat34Vec3f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using YannBench34d = BenchFixture<Vec4d, Vec4d, Mat34d>;
BENCHMARK_DEFINE_F(YannBench34d, Mat34Vec3d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      y[i].head<3>().noalias() = M[i] * x[i];
      y[i].coeffRef(3) = 1.0f;
    }
  }
}
BENCHMARK_REGISTER_F(YannBench34d, Mat34Vec3d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using DotBench3f = BenchFixture<Vec3f, Vec3f, Mat3f>;
BENCHMARK_DEFINE_F(DotBench3f, Vec3f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      dotprod[i] = x[i].dot(t[i]);
    }
  }
}
BENCHMARK_REGISTER_F(DotBench3f, Vec3f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using DotBench3d = BenchFixture<Vec3d, Vec3d, Mat3d>;
BENCHMARK_DEFINE_F(DotBench3d, Vec3d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      dotprod[i] = x[i].dot(t[i]);
    }
  }
}
BENCHMARK_REGISTER_F(DotBench3d, Vec3d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using DotBench4f = BenchFixture<Vec4f, Vec4f, Mat4f>;
BENCHMARK_DEFINE_F(DotBench4f, Vec4f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      dotprod[i] = x[i].dot(t[i]);
    }
  }
}
BENCHMARK_REGISTER_F(DotBench4f, Vec4f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using DotBench4d = BenchFixture<Vec4d, Vec4d, Mat4d>;
BENCHMARK_DEFINE_F(DotBench4d, Vec4d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      dotprod[i] = x[i].dot(t[i]);
    }
  }
}
BENCHMARK_REGISTER_F(DotBench4d, Vec4d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using YannDotBench4f = BenchFixture<Vec4f, Vec4f, Mat4f>;
BENCHMARK_DEFINE_F(YannDotBench4f, Vec4f)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      dotprod[i] = (x[i].cwiseProduct(t[i])).head<3>().sum();
    }
  }
}
BENCHMARK_REGISTER_F(YannDotBench4f, Vec4f)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

using YannDotBench4d = BenchFixture<Vec4d, Vec4d, Mat4d>;
BENCHMARK_DEFINE_F(YannDotBench4d, Vec4d)(benchmark::State & st)
{
  for (auto _ : st)
  {
    for (std::size_t i = 0; i < x.size(); ++i)
    {
      dotprod[i] = (x[i].cwiseProduct(t[i])).head<3>().sum();
    }
  }
}
BENCHMARK_REGISTER_F(YannDotBench4d, Vec4d)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(WARMUP_TIME)->MinTime(MIN_TIME);

BENCHMARK_MAIN();
