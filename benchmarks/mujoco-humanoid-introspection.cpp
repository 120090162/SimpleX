//
// Copyright (c) 2025 INRIA
//

#include "simple/core/simulator.hpp"
#include "tests/test_data/config.h"

#include <pinocchio/algorithm/fwd.hpp>
#include <pinocchio/parsers/mjcf.hpp>

#include <benchmark/benchmark.h>

using namespace simple;
using namespace pinocchio;
using ModelHandle = Simulator::ModelHandle;
using DataHandle = Simulator::DataHandle;
using GeometryModelHandle = Simulator::GeometryModelHandle;
using GeometryDataHandle = Simulator::GeometryDataHandle;
using ConstraintProblem = Simulator::ConstraintProblem;
using ConstraintModelVector = ConstraintProblem::WrappedConstraintModelVector;
using ConstraintDataVector = ConstraintProblem::WrappedConstraintDataVector;
#define ADMM ::pinocchio::ADMMConstraintSolverTpl
#define PGS ::pinocchio::PGSConstraintSolverTpl

using DelassusRigidBody = Simulator::DelassusRigidBodyOperator;
using DelassusCholesky = Simulator::DelassusCholeskyExpressionOperator;
using ConstraintCholeskyDecomposition = Simulator::ConstraintCholeskyDecomposition;

constexpr double damping_value = 1e-6;

using namespace ::pinocchio::helper;
template<typename DelassusOperator>
struct create_delassus_impl
{
  static DelassusOperator
  run(const Model & model, Data & data, const ConstraintModelVector & constraint_models, const ConstraintDataVector & constraint_datas);
};

template<>
struct create_delassus_impl<DelassusRigidBody>
{
  static DelassusRigidBody
  run(const Model & model, Data & data, const ConstraintModelVector & constraint_models, const ConstraintDataVector & constraint_datas)
  {
    DelassusRigidBody delassus(make_ref(model), make_ref(data), make_ref(constraint_models), make_ref(constraint_datas));
    return delassus;
  }
};

template<>
struct create_delassus_impl<ConstraintCholeskyDecomposition>
{
  static ConstraintCholeskyDecomposition
  run(const Model & model, Data & data, const ConstraintModelVector & constraint_models, const ConstraintDataVector & constraint_datas)
  {
    ConstraintCholeskyDecomposition cholesky(model, data, constraint_models, constraint_datas);
    return cholesky;
  }
};

template<>
struct create_delassus_impl<DelassusOperatorDense>
{
  // static ConstraintCholeskyDecomposition internal_cholesky;
  static DelassusOperatorDense
  run(const Model & model, Data & data, const ConstraintModelVector & constraint_models, const ConstraintDataVector & constraint_datas)
  {
    ConstraintCholeskyDecomposition internal_cholesky(model, data, constraint_models, constraint_datas);
    // const Eigen::DenseIndex total_dim = internal_cholesky.size();
    // const Eigen::DenseIndex constraint_dim = total_dim - model.nv;
    internal_cholesky.compute(model, data, constraint_models, constraint_datas, damping_value);
    DelassusOperatorDense delassus = DelassusOperatorDense(internal_cholesky.getDelassusCholeskyExpression().dense());
    // DelassusOperatorDense delassus = DelassusOperatorDense(Eigen::MatrixXd::Zero(constraint_dim, constraint_dim));
    return delassus;
  }
};

template<typename DelassusOperator>
DelassusOperator create_delassus(
  const Model & model, Data & data, const ConstraintModelVector & constraint_models, const ConstraintDataVector & constraint_datas)
{
  return create_delassus_impl<DelassusOperator>::run(model, data, constraint_models, constraint_datas);
}

template<typename DelassusOperator>
struct compute_delassus_impl
{
  template<typename ConfigVector>
  static void run(
    DelassusOperator & delassus,
    const Model & model,
    Data & data,
    const ConstraintModelVector & constraint_models,
    ConstraintDataVector & constraint_datas,
    const Eigen::MatrixBase<ConfigVector> & q);
};

template<>
struct compute_delassus_impl<DelassusRigidBody>
{
  template<typename ConfigVector>
  static void run(
    DelassusRigidBody & delassus,
    const Model & model,
    Data & data,
    const ConstraintModelVector & constraint_models,
    ConstraintDataVector & constraint_datas,
    const Eigen::MatrixBase<ConfigVector> & q)
  {
    PINOCCHIO_UNUSED_VARIABLE(model);
    PINOCCHIO_UNUSED_VARIABLE(data);
    PINOCCHIO_UNUSED_VARIABLE(constraint_models);
    PINOCCHIO_UNUSED_VARIABLE(constraint_datas);
    PINOCCHIO_UNUSED_VARIABLE(q);
    delassus.updateDamping(damping_value);
    delassus.compute();
  }
};

bool use_explicit_delassus = true;

template<>
struct compute_delassus_impl<ConstraintCholeskyDecomposition>
{
  template<typename ConfigVector>
  static void run(
    ConstraintCholeskyDecomposition & delassus,
    const Model & model,
    Data & data,
    const ConstraintModelVector & constraint_models,
    ConstraintDataVector & constraint_datas,
    const Eigen::MatrixBase<ConfigVector> & q)
  {
    // crba needs to be taken into account because there is no need for crba in rigid body delassus
    crba(model, data, q, ::pinocchio::Convention::WORLD);
    delassus.compute(model, data, constraint_models, constraint_datas, damping_value, use_explicit_delassus);
  }
};

template<>
struct compute_delassus_impl<DelassusOperatorDense>
{
  template<typename ConfigVector>
  static void run(
    DelassusOperatorDense & delassus,
    const Model & model,
    Data & data,
    const ConstraintModelVector & constraint_models,
    ConstraintDataVector & constraint_datas,
    const Eigen::MatrixBase<ConfigVector> & q)
  {
    // crba needs to be taken into account because there is no need for crba in rigid body delassus
    crba(model, data, q, ::pinocchio::Convention::WORLD);
    // ConstraintCholeskyDecomposition cholesky(model, data, constraint_models, constraint_datas);
    // cholesky.compute(model, data, constraint_models, constraint_datas, damping_value);

    // delassus.compute(model, data, constraint_models, constraint_datas, damping_value);
  }
};

template<typename DelassusOperator, typename ConfigVector>
void compute_delassus(
  DelassusOperator & delassus,
  const Model & model,
  Data & data,
  const ConstraintModelVector & constraint_models,
  ConstraintDataVector & constraint_datas,
  const Eigen::MatrixBase<ConfigVector> & q)
{
  compute_delassus_impl<DelassusOperator>::run(delassus, model, data, constraint_models, constraint_datas, q.derived());
}

template<typename DelassusOperator>
struct update_delassus_damping_impl
{
  template<typename Scalar>
  static void run(DelassusOperator & delassus, const Scalar value);
};

template<>
struct update_delassus_damping_impl<DelassusRigidBody>
{
  template<typename Scalar>
  static void run(DelassusRigidBody & delassus, const Scalar value)
  {
    delassus.updateDamping(value);
    delassus.compute(false, true);
  }
};

template<>
struct update_delassus_damping_impl<ConstraintCholeskyDecomposition>
{
  template<typename Scalar>
  static void run(ConstraintCholeskyDecomposition & delassus, const Scalar value)
  {
    delassus.updateDamping(value, use_explicit_delassus);
  }
};

template<>
struct update_delassus_damping_impl<DelassusOperatorDense>
{
  template<typename Scalar>
  static void run(DelassusOperatorDense & delassus, const Scalar value)
  {
    delassus.updateDamping(value);
    delassus.runProtectedCholeskyDecomposition();
  }
};

template<typename DelassusOperator, typename Scalar>
void update_delassus_damping(DelassusOperator & delassus, const Scalar value)
{
  update_delassus_damping_impl<DelassusOperator>::run(delassus, value);
}

template<typename DelassusOperator>
struct apply_on_the_right_impl
{
  template<typename Scalar>
  static void run(DelassusOperator & delassus, const Scalar value);
};

template<>
struct apply_on_the_right_impl<DelassusRigidBody>
{
  template<typename InputVector, typename OutputVector>
  static void run(
    DelassusRigidBody & delassus,
    const Eigen::MatrixBase<InputVector> & input_vector,
    const Eigen::MatrixBase<OutputVector> & output_vector)
  {
    delassus.applyOnTheRight(input_vector, output_vector.const_cast_derived());
  }
};

template<>
struct apply_on_the_right_impl<DelassusOperatorDense>
{
  template<typename InputVector, typename OutputVector>
  static void run(
    DelassusOperatorDense & delassus,
    const Eigen::MatrixBase<InputVector> & input_vector,
    const Eigen::MatrixBase<OutputVector> & output_vector)
  {
    delassus.applyOnTheRight(input_vector, output_vector.const_cast_derived());
  }
};

template<>
struct apply_on_the_right_impl<ConstraintCholeskyDecomposition>
{
  template<typename InputVector, typename OutputVector>
  static void run(
    ConstraintCholeskyDecomposition & delassus,
    const Eigen::MatrixBase<InputVector> & input_vector,
    const Eigen::MatrixBase<OutputVector> & output_vector)
  {
    delassus.getDelassusCholeskyExpression().applyOnTheRight(input_vector, output_vector.const_cast_derived(), use_explicit_delassus);
  }
};

template<typename DelassusOperator, typename InputVector, typename OutputVector>
void apply_on_the_right(
  DelassusOperator & delassus, const Eigen::MatrixBase<InputVector> & input_vector, const Eigen::MatrixBase<OutputVector> & output_vector)
{
  apply_on_the_right_impl<DelassusOperator>::run(delassus, input_vector, output_vector.const_cast_derived());
}

template<typename DelassusOperator>
struct solve_in_place_impl
{
  template<typename InputVector>
  static void run(DelassusOperator & delassus, const Eigen::MatrixBase<InputVector> & input_vector);
};

template<>
struct solve_in_place_impl<DelassusRigidBody>
{
  template<typename InputVector>
  static void run(DelassusRigidBody & delassus, const Eigen::MatrixBase<InputVector> & input_vector)
  {
    delassus.solveInPlace(input_vector.const_cast_derived());
  }
};

template<>
struct solve_in_place_impl<ConstraintCholeskyDecomposition>
{
  template<typename InputVector>
  static void run(ConstraintCholeskyDecomposition & delassus, const Eigen::MatrixBase<InputVector> & input_vector)
  {
    delassus.getDelassusCholeskyExpression().solveInPlace(input_vector.const_cast_derived());
  }
};

template<>
struct solve_in_place_impl<DelassusOperatorDense>
{
  template<typename InputVector>
  static void run(DelassusOperatorDense & delassus, const Eigen::MatrixBase<InputVector> & input_vector)
  {
    delassus.solveInPlace(input_vector.const_cast_derived());
  }
};

template<typename DelassusOperator, typename InputVector>
void solve_in_place(DelassusOperator & delassus, const Eigen::MatrixBase<InputVector> & input_vector)
{
  solve_in_place_impl<DelassusOperator>::run(delassus, input_vector.const_cast_derived());
}

const double DT = 1e-3;

void addFloorToGeomModel(GeometryModel & geom_model)
{
  using CollisionGeometryPtr = GeometryObject::CollisionGeometryPtr;

  CollisionGeometryPtr floor_collision_shape = CollisionGeometryPtr(new hpp::fcl::Halfspace(0.0, 0.0, 1.0, 0.0));

  const SE3 M = SE3::Identity();
  GeometryObject floor("floor", 0, 0, M, floor_collision_shape);
  geom_model.addGeometryObject(floor);
}

void addSystemCollisionPairs(const Model & model, GeometryModel & geom_model, const Eigen::VectorXd & qref)
{
  Data data(model);
  GeometryData geom_data(geom_model);
  // TI this function to gain compilation speed on this test
  ::pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data, qref);
  geom_model.removeAllCollisionPairs();
  for (std::size_t i = 0; i < geom_model.geometryObjects.size(); ++i)
  {
    for (std::size_t j = i; j < geom_model.geometryObjects.size(); ++j)
    {
      if (i == j)
      {
        continue; // don't add collision pair if same object
      }
      const GeometryObject & gobj_i = geom_model.geometryObjects[i];
      const GeometryObject & gobj_j = geom_model.geometryObjects[j];
      if (gobj_i.name == "floor" || gobj_j.name == "floor")
      { // if floor, always add a collision pair
        geom_model.addCollisionPair(CollisionPair(i, j));
      }
      else
      {
        if (gobj_i.parentJoint == gobj_j.parentJoint)
        { // don't add collision pair if same parent
          continue;
        }

        // run collision detection -- add collision pair if shapes are not colliding
        const SE3 M1 = geom_data.oMg[i];
        const SE3 M2 = geom_data.oMg[j];

        hpp::fcl::CollisionRequest colreq;
        colreq.security_margin = 1e-2; // 1cm of clearance
        hpp::fcl::CollisionResult colres;
        hpp::fcl::collide(
          gobj_i.geometry.get(), ::pinocchio::toFclTransform3f(M1), //
          gobj_j.geometry.get(), ::pinocchio::toFclTransform3f(M2), //
          colreq, colres);
        if (!colres.isCollision())
        {
          geom_model.addCollisionPair(CollisionPair(i, j));
        }
      }
    }
  }
}

struct SimulationRunner
{
  SimulationRunner()
  {
    init();
  }

  SimulationRunner(SimulationRunner && other)
  : sim_ptr(std::move(other.sim_ptr))
  , q0(std::move(other.q0))
  , v0(std::move(other.v0))
  , zero_torque(std::move(other.zero_torque))
  {
  }

  void init()
  {
    // std::cout << "SimulationRunner::init" << std::endl;
    ModelHandle model_handle = std::make_shared<Model>();
    Model & model = ::pinocchio::helper::get_ref(model_handle);
    GeometryModelHandle geom_model_handle = std::make_shared<GeometryModel>();
    GeometryModel & geom_model = ::pinocchio::helper::get_ref(geom_model_handle);

    const bool verbose = false;
    ::pinocchio::mjcf::buildModel(SIMPLE_TEST_DATA_DIR "/mujoco_humanoid.xml", model, verbose);
    ::pinocchio::mjcf::buildGeom(model, SIMPLE_TEST_DATA_DIR "/mujoco_humanoid.xml", pinocchio::COLLISION, geom_model);
    addFloorToGeomModel(geom_model);

    // initial state
    q0 = model.referenceConfigurations["qpos0"];
    v0 = Eigen::VectorXd::Zero(model.nv);
    zero_torque = Eigen::VectorXd::Zero(model.nv);

    // add collision pairs
    addSystemCollisionPairs(model, geom_model, q0);

    // tmp: remove joint friction
    model.lowerDryFrictionLimit.tail(model.nv - 6).setOnes();
    model.upperDryFrictionLimit.tail(model.nv - 6).setOnes();

    // tmp: remove joint limits
    model.lowerPositionLimit.setConstant(-std::numeric_limits<double>::infinity());
    model.upperPositionLimit.setConstant(std::numeric_limits<double>::infinity());

    sim_ptr = std::make_unique<Simulator>(model_handle, geom_model_handle);
  }

  template<template<typename> class ConstraintSolver>
  void run(const int num_steps, const bool warm_start = true)
  {
    // std::cout << "----------" << std::endl;
    Eigen::VectorXd q = q0;
    Eigen::VectorXd v = v0;

    auto & sim = get_ref(sim_ptr);

    sim.config.warmstart_constraint_velocities = warm_start;

    for (int i = 0; i < num_steps; ++i)
    {
      sim.step<ConstraintSolver>(q, v, zero_torque, DT);
      q = sim.state.qnew;
      v = sim.state.vnew;
    }
  }

  void stats() const
  {
    const auto & sim = get_ref(sim_ptr);
    const Model & model = sim.model();
    const ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
    const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
    const ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;
    const auto num_constraints = pinocchio::getTotalConstraintResidualSize(constraint_models, constraint_datas);

    std::cout << "simulation stats:" << std::endl;
    ;
    std::cout << "  - nq: " << model.nq << std::endl;
    std::cout << "  - nv: " << model.nv << std::endl;
    std::cout << "  - joint friction: " << model.lowerDryFrictionLimit.transpose() << std::endl;
    std::cout << "  - constraint_models.size(): " << constraint_models.size() << std::endl;
    std::cout << "  - num_constraints: " << num_constraints << std::endl;
    std::cout << "  - num it: " << sim.workspace.constraint_solvers.admm_solver.getIterationCount() << std::endl;
    std::cout << "  - max it: " << sim.workspace.constraint_solvers.admm_solver.getMaxIterations() << std::endl;
    std::cout << "  - abs prec: " << sim.workspace.constraint_solvers.admm_solver.getAbsolutePrecision() << std::endl;
    // std::cout << "model:\n" << model << std::endl;
    std::cout << "----------" << std::endl;
  }

  Simulator & sim()
  {
    return get_ref(sim_ptr);
  }
  const Simulator & sim() const
  {
    return get_ref(sim_ptr);
  }

  std::unique_ptr<Simulator> sim_ptr;
  Eigen::VectorXd q0;
  Eigen::VectorXd v0;
  Eigen::VectorXd zero_torque;
};

struct MujocoHumanoidFixture : benchmark::Fixture
{
  void SetUp(benchmark::State &)
  {
    SimulationRunner sim_runner;
    sim_runner.run<ADMM>(2000, true);
    sim_runner.stats();
    sim_runner_ptr = std::make_unique<SimulationRunner>(std::move(sim_runner));
  }

  void TearDown(benchmark::State &)
  {
  }

  std::unique_ptr<SimulationRunner> sim_runner_ptr;
};

BENCHMARK_DEFINE_F(MujocoHumanoidFixture, create_delassus)(benchmark::State & st)
{

  auto & sim = get_ref(sim_runner_ptr->sim_ptr);
  const Model & model = sim.model();
  Data & data = sim.data();
  const ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
  const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
  const ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;
  for (auto _ : st)
  {
    auto delassus = create_delassus<DelassusRigidBody>(model, data, constraint_models, constraint_datas);
    benchmark::DoNotOptimize(delassus);
    // sim_ptr->step<ADMM>(sim_ptr->qnew, sim_ptr->vnew, zero_torque, DT);
  }
}

// BENCHMARK_REGISTER_F(MujocoHumanoidFixture, create_delassus)
//   ->Unit(benchmark::kMicrosecond)
//   ->MinWarmUpTime(3.)
//   // ->MinTime(5.)
//   ;

template<typename DelassusOperator>
void create_delassus_b(benchmark::State & st)
{
  SimulationRunner sim_instance;

  sim_instance.run<ADMM>(2000, true);
  // sim_instance.stats();

  auto sim = sim_instance.sim();
  const Model & model = sim.model();
  Data & data = sim.data();
  ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
  const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
  ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;

  for (auto _ : st)
  {
    auto delassus = create_delassus<DelassusOperator>(model, data, constraint_models, constraint_datas);
    benchmark::DoNotOptimize(delassus);
    // sim_ptr->step<ADMM>(sim_ptr->qnew, sim_ptr->vnew, zero_torque, DT);
  }
}

BENCHMARK(create_delassus_b<DelassusRigidBody>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(create_delassus_b<ConstraintCholeskyDecomposition>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(create_delassus_b<DelassusOperatorDense>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);

template<typename DelassusOperator>
void compute_delassus_b(benchmark::State & st)
{
  SimulationRunner sim_instance;

  sim_instance.run<ADMM>(2000, true);
  // sim_instance.stats();

  auto sim = sim_instance.sim();
  const Model & model = sim.model();
  Data & data = sim.data();
  ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
  const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
  ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;
  auto delassus = create_delassus<DelassusOperator>(model, data, constraint_models, constraint_datas);
  for (auto _ : st)
  {
    compute_delassus(delassus, model, data, constraint_models, constraint_datas, sim.state.q);
    benchmark::DoNotOptimize(delassus);
    // sim_ptr->step<ADMM>(sim_ptr->qnew, sim_ptr->vnew, zero_torque, DT);
  }
}

BENCHMARK(compute_delassus_b<DelassusRigidBody>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(compute_delassus_b<ConstraintCholeskyDecomposition>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);

template<typename DelassusOperator>
void update_damping_b(benchmark::State & st)
{
  SimulationRunner sim_instance;
  sim_instance.run<ADMM>(2000, true);
  // sim_instance.stats();

  auto sim = sim_instance.sim();
  const Model & model = sim.model();
  Data & data = sim.data();
  ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
  const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
  ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;
  auto delassus = create_delassus<DelassusOperator>(model, data, constraint_models, constraint_datas);
  compute_delassus(delassus, model, data, constraint_models, constraint_datas, sim.state.q);
  for (auto _ : st)
  {
    update_delassus_damping(delassus, 1e-2);
    benchmark::DoNotOptimize(delassus);
    // sim_ptr->step<ADMM>(sim_ptr->qnew, sim_ptr->vnew, zero_torque, DT);
  }
}

BENCHMARK(update_damping_b<DelassusRigidBody>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(update_damping_b<ConstraintCholeskyDecomposition>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(update_damping_b<DelassusOperatorDense>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);

template<typename DelassusOperator>
void apply_on_the_right_b(benchmark::State & st)
{
  SimulationRunner sim_instance;
  sim_instance.run<ADMM>(2000, true);
  // sim_instance.stats();

  auto sim = sim_instance.sim();
  const Model & model = sim.model();
  Data & data = sim.data();
  ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
  const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
  ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;
  auto delassus = create_delassus<DelassusOperator>(model, data, constraint_models, constraint_datas);
  compute_delassus(delassus, model, data, constraint_models, constraint_datas, sim.state.q);

  const auto size = pinocchio::getTotalConstraintResidualSize(constraint_models, constraint_datas);
  const Eigen::VectorXd vec_in = Eigen::VectorXd::Random(size);
  Eigen::VectorXd vec_out = Eigen::VectorXd::Zero(size);
  for (auto _ : st)
  {
    apply_on_the_right(delassus, vec_in, vec_out);
    benchmark::DoNotOptimize(vec_out);
  }
}

BENCHMARK(apply_on_the_right_b<DelassusRigidBody>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(apply_on_the_right_b<ConstraintCholeskyDecomposition>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);
BENCHMARK(apply_on_the_right_b<DelassusOperatorDense>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(10000);

template<typename DelassusOperator>
void solve_in_place_b(benchmark::State & st)
{
  SimulationRunner sim_instance;
  sim_instance.run<ADMM>(2000, true);
  // sim_instance.stats();

  auto sim = sim_instance.sim();
  const Model & model = sim.model();
  Data & data = sim.data();
  ConstraintProblem & constraint_problem = sim.workspace.constraint_problem();
  const ConstraintModelVector & constraint_models = constraint_problem.constraint_models;
  ConstraintDataVector & constraint_datas = constraint_problem.constraint_datas;
  auto delassus = create_delassus<DelassusOperator>(model, data, constraint_models, constraint_datas);
  compute_delassus(delassus, model, data, constraint_models, constraint_datas, sim.state.q);

  const auto size = pinocchio::getTotalConstraintResidualSize(constraint_models, constraint_datas);
  const Eigen::VectorXd vec_in = Eigen::VectorXd::Random(size);
  Eigen::VectorXd vec_out = Eigen::VectorXd::Zero(size);
  solve_in_place(delassus, vec_out);
  for (auto _ : st)
  {
    vec_out = vec_in;
    solve_in_place(delassus, vec_out);
    benchmark::DoNotOptimize(vec_out);
  }
}

BENCHMARK(solve_in_place_b<DelassusRigidBody>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(100000);
BENCHMARK(solve_in_place_b<ConstraintCholeskyDecomposition>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(100000);
BENCHMARK(solve_in_place_b<DelassusOperatorDense>)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(100000);

void run_last_simulation_step_b(benchmark::State & st)
{
  SimulationRunner sim_instance;
  sim_instance.run<ADMM>(2000, true);
  sim_instance.stats();

  auto sim = sim_instance.sim();

  const Eigen::VectorXd q = sim.state.qnew;
  const Eigen::VectorXd v = sim.state.vnew;

  for (auto _ : st)
  {
    sim.step<ADMM>(q, v, sim_instance.zero_torque, DT);
    benchmark::DoNotOptimize(sim.state.qnew);
    benchmark::DoNotOptimize(sim.state.vnew);
  }
}

BENCHMARK(run_last_simulation_step_b)->Unit(benchmark::kMicrosecond)->MinWarmUpTime(1.)->Iterations(1000);

void run_cholesly_b(benchmark::State & st)
{
  const auto size = st.range(0);
  typedef Eigen::MatrixXd Matrix;

  Matrix mat_in = Matrix::Identity(size, size);

  Eigen::LDLT<Matrix> ldlt(mat_in);
  for (auto _ : st)
  {
    ldlt.compute(mat_in);
    benchmark::DoNotOptimize(ldlt);
  }
}

BENCHMARK(run_cholesly_b)
  ->Unit(benchmark::kMicrosecond)
  ->MinWarmUpTime(1.)
  ->Arg(10)
  ->Arg(20)
  ->Arg(30)
  ->Arg(40)
  ->Arg(50)
  ->Arg(60)
  ->Arg(100)
  ->Arg(120)
  ->Arg(150)
  ->Arg(200)
  ->Arg(250)
  ->Arg(300)
  ->Iterations(10000);

BENCHMARK_MAIN();
