#include "simplex/core/simulator-x.hpp"
#include "../tests/test-utils.hpp"

#include <pinocchio/algorithm/fwd.hpp>
#include <pinocchio/parsers/mjcf.hpp>

#include <benchmark/benchmark.h>

using namespace simplex;
using namespace pinocchio;
using ModelHandle = SimulatorX::ModelHandle;
using DataHandle = SimulatorX::DataHandle;
using GeometryModelHandle = SimulatorX::GeometryModelHandle;
using GeometryDataHandle = SimulatorX::GeometryDataHandle;
#define ADMM ::simplex::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl

const double DT = 1e-3;
const std::size_t NUM_SIM_STEPS = 100;

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

struct MujocoHumanoidFixture : benchmark::Fixture
{
    void SetUp(benchmark::State &)
    {
        ModelHandle model_handle = std::make_shared<Model>();
        GeometryModelHandle geom_model_handle = std::make_shared<GeometryModel>();
        Model & model = ::pinocchio::helper::get_ref(model_handle);
        GeometryModel & geom_model = ::pinocchio::helper::get_ref(geom_model_handle);

        const bool verbose = false;
        auto mjcf_path = findTestResource("SIMPLEX/tests/resources/mujoco_humanoid.xml");
        ::pinocchio::mjcf::buildModel(mjcf_path, model, verbose);
        ::pinocchio::mjcf::buildGeom(model, mjcf_path, pinocchio::COLLISION, geom_model);
        addFloorToGeomModel(geom_model);

        // initial state
        q0 = model.referenceConfigurations["qpos0"];
        v0 = Eigen::VectorXd::Zero(model.nv);
        zero_torque = Eigen::VectorXd::Zero(model.nv);

        // add collision pairs
        addSystemCollisionPairs(model, geom_model, q0);

        sim = std::make_unique<SimulatorX>(model_handle, geom_model_handle);
    }

    void TearDown(benchmark::State &)
    {
    }

    std::unique_ptr<SimulatorX> sim;
    Eigen::VectorXd q0;
    Eigen::VectorXd v0;
    Eigen::VectorXd zero_torque;
};

BENCHMARK_DEFINE_F(MujocoHumanoidFixture, admm)(benchmark::State & st)
{
    for (auto _ : st)
    {
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        sim->reset();
        bool warm_start = st.range(0);
        sim->config.warmstart_constraint_velocities = warm_start;
        for (size_t i = 0; i < NUM_SIM_STEPS; ++i)
        {
            sim->step<ADMM>(q, v, zero_torque, DT);
            q = sim->state.qnew;
            v = sim->state.vnew;
        }
    }
}
BENCHMARK_REGISTER_F(MujocoHumanoidFixture, admm)
    ->ArgName("warmstart")
    ->Arg(0)
    ->Arg(1)
    ->Unit(benchmark::kMillisecond)
    ->MinWarmUpTime(3.)
    ->MinTime(5.);

BENCHMARK_DEFINE_F(MujocoHumanoidFixture, pgs)(benchmark::State & st)
{
    for (auto _ : st)
    {
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        sim->reset();
        bool warm_start = st.range(0);
        sim->config.warmstart_constraint_velocities = warm_start;
        for (size_t i = 0; i < NUM_SIM_STEPS; ++i)
        {
            sim->step<PGS>(q, v, zero_torque, DT);
            q = sim->state.qnew;
            v = sim->state.vnew;
        }
    }
}
BENCHMARK_REGISTER_F(MujocoHumanoidFixture, pgs)
    ->ArgName("warmstart")
    ->Arg(0)
    ->Arg(1)
    ->Unit(benchmark::kMillisecond)
    ->MinWarmUpTime(3.)
    ->MinTime(5.);

BENCHMARK_MAIN();
