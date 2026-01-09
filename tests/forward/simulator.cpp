#include "simplex/core/simulator.hpp"
#include <pinocchio/algorithm/fwd.hpp>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/parsers/mjcf.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <memory>

#include "../test-utils.hpp"

using namespace simplex;
using namespace pinocchio;
using ModelHandle = Simulator::ModelHandle;
using DataHandle = Simulator::DataHandle;
using GeometryModelHandle = Simulator::GeometryModelHandle;
using GeometryDataHandle = Simulator::GeometryDataHandle;
using BilateralPointConstraintModel = Simulator::BilateralPointConstraintModel;
using BilateralPointConstraintModelVector = Simulator::BilateralPointConstraintModelVector;
using WeldConstraintModel = Simulator::WeldConstraintModel;
using WeldConstraintModelVector = Simulator::WeldConstraintModelVector;

#define ADMM ::pinocchio::ADMMContactSolverTpl

using ModelHandle = Simulator::ModelHandle;
using GeometryModelHandle = Simulator::GeometryModelHandle;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(simulator_instance_constructor)
{
    ModelHandle model(new Model());
    DataHandle data(new Data(*model));
    GeometryModelHandle geom_model(new GeometryModel());
    GeometryDataHandle geom_data(new GeometryData(*geom_model));
    BOOST_CHECK_NO_THROW(Simulator sim(model, data, geom_model, geom_data));
}

BOOST_AUTO_TEST_CASE(simulator_instance_constructor_2)
{
    ModelHandle model(new Model());
    GeometryModelHandle geom_model(new GeometryModel());
    BOOST_CHECK_NO_THROW(Simulator sim(model, geom_model));
}

BOOST_AUTO_TEST_CASE(simulator_instance_step)
{
    ModelHandle model(new Model());
    buildModels::humanoidRandom(*model);
    buildModels::manipulator(*model);

    GeometryModelHandle geom_model(new GeometryModel());
    buildModels::manipulatorGeometries(*model, *geom_model);

    Simulator sim(model, geom_model);

    Eigen::VectorXd q = neutral(*model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
    INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
}

BOOST_AUTO_TEST_CASE(simulator_instance_step_make_shared)
{
    Model _model;
    buildModels::manipulator(_model);
    ModelHandle model = std::make_shared<Model>(_model);

    GeometryModel _geom_model;
    buildModels::manipulatorGeometries(_model, _geom_model);
    GeometryModelHandle geom_model = std::make_shared<GeometryModel>(_geom_model);
    Simulator sim(model, geom_model);

    Eigen::VectorXd q = neutral(*model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
    INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
}

BOOST_AUTO_TEST_CASE(simulator_instance_step_cubes)
{
    const int N_cubes = 10;
    std::vector<double> radius(N_cubes, 1.0);
    Model _model;
    GeometryModel _geom_model;
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const FrameIndex plane_frame = FrameIndex(0);
    const GeometryObject plane_geom = GeometryObject("plane", 0, plane_frame, SE3::Identity(), plane_ptr);
    GeomIndex plane_id = _geom_model.addGeometryObject(plane_geom);
    for (int i = 0; i < N_cubes; ++i)
    {
        const std::string name = "cube_" + std::to_string(i);
        const std::string joint_name = "joint_" + std::to_string(i);
        const std::string frame_name = "frame_" + std::to_string(i);
        const double mass = 1.0;
        const double r = radius[static_cast<std::size_t>(i)];
        const Inertia inertia = Inertia::FromBox(mass, r, r, r);
        const JointModelFreeFlyer joint = JointModelFreeFlyer();
        const JointIndex parent = 0;
        const SE3 placement = SE3::Identity();
        JointIndex joint_id = _model.addJoint(parent, joint, placement, joint_name);
        _model.appendBodyToJoint(joint_id, inertia);
        coal::CollisionGeometryPtr_t box_ptr(new coal::Box(r, r, r));
        const GeometryObject box_geom = GeometryObject(name, joint_id, placement, box_ptr);
        GeomIndex box_id = _geom_model.addGeometryObject(box_geom);
        CollisionPair cp(plane_id, box_id);
        _geom_model.addCollisionPair(cp);
    }

    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    Simulator sim(model, geom_model);
    sim.admm_constraint_solver_settings.absolute_precision = 1e-12;
    sim.admm_constraint_solver_settings.relative_precision = 1e-12;

    Eigen::VectorXd q = neutral(*model);
    for (int i = 0; i < N_cubes; ++i)
    {
        q(i * 7) = static_cast<double>(i) * radius[static_cast<std::size_t>(i)] * 1.1;
        q(i * 7 + 2) = radius[static_cast<std::size_t>(i)] / 2.;
    }
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);
    INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
    // 4 contact points (cube-plane)
    INDEX_EQUALITY_CHECK(4 * N_cubes, sim.constraints_problem().constraint_models.size());
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
    // Calling a second time step to test warmstart
    q = sim.qnew;
    v = sim.vnew;
    sim.step<ADMM>(q, v, tau, dt);
    // TODO: TEST - It gives 2 instead of 0 :( Warmstart must be solved
    INDEX_INEQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 2);
    INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
    INDEX_EQUALITY_CHECK(4 * N_cubes, sim.constraints_problem().constraint_models.size());
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
}

BOOST_AUTO_TEST_CASE(simulator_instance_checkCollisionPairs)
{
    ModelHandle model(new Model());
    GeometryModelHandle geom_model(new GeometryModel());

    // Add balls
    const double radius = 1.0;
    const double mass = 1.0;
    const std::string name1 = "ball_1";
    const std::string name2 = "ball_1";
    const std::string joint_name = "joint_1";
    const std::string frame_name = "frame_1";
    const Inertia inertia = Inertia::FromSphere(mass, radius);
    const JointModelFreeFlyer joint = JointModelFreeFlyer();
    const JointIndex parent = 0;
    const SE3 placement = SE3::Identity();
    JointIndex joint_id = model->addJoint(parent, joint, placement, joint_name);
    model->appendBodyToJoint(joint_id, inertia);
    coal::CollisionGeometryPtr_t ball_ptr(new coal::Sphere(radius));
    const GeometryObject ball_geom1 = GeometryObject(name1, joint_id, placement, ball_ptr);
    const GeometryObject ball_geom2 = GeometryObject(name2, joint_id, placement, ball_ptr);
    GeomIndex ball_id1 = geom_model->addGeometryObject(ball_geom1);
    GeomIndex ball_id2 = geom_model->addGeometryObject(ball_geom2);

    CollisionPair col_pair(ball_id1, ball_id2);
    geom_model->addCollisionPair(col_pair);

    Simulator sim(model, geom_model);
    BOOST_CHECK(!sim.checkCollisionPairs());
}

BOOST_AUTO_TEST_SUITE_END()
