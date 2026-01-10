#include "simplex/core/simulator-x.hpp"

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
using ModelHandle = SimulatorX::ModelHandle;
using DataHandle = SimulatorX::DataHandle;
using GeometryModelHandle = SimulatorX::GeometryModelHandle;
using GeometryDataHandle = SimulatorX::GeometryDataHandle;
using BilateralPointConstraintModel = SimulatorX::BilateralPointConstraintModel;
using BilateralPointConstraintModelVector = SimulatorX::BilateralPointConstraintModelVector;

using VectorXd = Eigen::VectorXd;

#define ADMM ::simplex::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl
#define Clarabel ::simplex::ClarabelContactSolverTpl

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(humanoid_with_point_anchor_constraint)
{
    // Construct humanoid
    ModelHandle model(new pin::Model());
    pin::buildModels::humanoid(*model, true);
    model->lowerPositionLimit = VectorXd::Constant(model->nq, -1.0) * std::numeric_limits<double>::max();
    model->upperPositionLimit = VectorXd::Constant(model->nq, 1.0) * std::numeric_limits<double>::max();
    DataHandle data = std::make_shared<pin::Data>(*model);

    GeometryModelHandle geom_model(new pin::GeometryModel());

    // Initial state
    VectorXd q = pinocchio::neutral(*model);

    // Point anchor constraint on the robot right wrist
    BilateralPointConstraintModelVector point_anchor_constraint_models;
    pinocchio::framesForwardKinematics(*model, *data, q);
    const pin::JointIndex joint1_id = 0;
    const pin::GeomIndex joint2_id = 13;
    ::pinocchio::SE3 Mc = data->oMi[joint2_id];
    const pin::SE3 joint1_placement = Mc;
    const pin::SE3 joint2_placement = pin::SE3::Identity();
    point_anchor_constraint_models.push_back(
        BilateralPointConstraintModel(*model, joint1_id, joint1_placement, joint2_id, joint2_placement));
    point_anchor_constraint_models[0].baumgarte_corrector_parameters().Kp = 0.1;

    // The humanoid's freeflyer's height should remain the same
    SimulatorX sim(model, geom_model);
    sim.addPointAnchorConstraints(point_anchor_constraint_models);
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-9;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-9;
    sim.config.constraint_solvers_configs.admm_config.max_iter = 1000;
    const double dt = 1e-3;
    VectorXd v = VectorXd::Zero(model->nv);
    VectorXd tau = VectorXd::Zero(model->nv);
    BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
    pinocchio::framesForwardKinematics(*model, *data, sim.state.qnew);
    EIGEN_VECTOR_IS_APPROX(Mc.translation(), data->oMi[joint2_id].translation(), 1e-3);

    // Calling the simulator twice to test warmstart
    sim.step<ADMM>(q, v, tau, dt);
    INDEX_EQUALITY_CHECK(sim.workspace.constraint_solvers.admm_solver.getIterationCount(), 0);

    for (int i = 0; i < 10; ++i)
    {
        VectorXd q = sim.state.qnew;
        VectorXd v = sim.state.vnew;
        BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
        pinocchio::framesForwardKinematics(*model, *data, sim.state.qnew);
        EIGEN_VECTOR_IS_APPROX(Mc.translation(), data->oMi[joint2_id].translation(), 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(simulator_instance_step_with_friction_on_joints)
{
    ModelHandle model(new pin::Model());
    pin::buildModels::manipulator(*model);
    model->lowerDryFrictionLimit = VectorXd::Ones(model->nv) * -100000.0;
    model->upperDryFrictionLimit = VectorXd::Ones(model->nv) * 100000.0;

    GeometryModelHandle geom_model(new pin::GeometryModel());
    pin::buildModels::manipulatorGeometries(*model, *geom_model);

    SimulatorX sim(model, geom_model);

    VectorXd q = neutral(*model);
    VectorXd v = VectorXd::Zero(model->nv);
    VectorXd tau = VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.state.vnew, VectorXd::Zero(model->nv), 1e-6);
    // calling the simulator again should not change the state
    // and the solver should converge in one iteration
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.state.vnew, VectorXd::Zero(model->nv), 1e-6);
    EIGEN_VECTOR_IS_APPROX(sim.state.vnew, v, 1e-6);
    EIGEN_VECTOR_IS_APPROX(sim.state.qnew, q, 1e-6);
    INDEX_EQUALITY_CHECK(sim.workspace.constraint_solvers.admm_solver.getIterationCount(), 0);
}

BOOST_AUTO_TEST_CASE(simulator_instance_step_with_limits_on_joints_for_manipulator)
{
    ModelHandle model(new pin::Model());
    pin::buildModels::manipulator(*model);
    // We first consider not active limits
    model->lowerPositionLimit = VectorXd::Ones(model->nq) * -100000.0;
    model->upperPositionLimit = VectorXd::Ones(model->nq) * 100000.0;

    GeometryModelHandle geom_model(new pin::GeometryModel());
    pin::buildModels::manipulatorGeometries(*model, *geom_model);

    SimulatorX sim(model, geom_model);

    VectorXd q = neutral(*model);
    VectorXd v = VectorXd::Zero(model->nv);
    VectorXd tau = VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);
    // simulated velocity should be the same as the one computed with aba as constraints are not active
    VectorXd vnew = v + dt * pinocchio::aba(*model, sim.data(), q, v, tau);
    EIGEN_VECTOR_IS_APPROX(sim.state.vnew, vnew, 1e-6);
    // calling the simulator again should not change the state
    // and the solver should converge in one iteration
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.state.vnew, vnew, 1e-6);
    INDEX_EQUALITY_CHECK(sim.workspace.constraint_solvers.admm_solver.getIterationCount(), 0);

    // we now consider active limits
    q = VectorXd::Zero(model->nv);
    model->lowerPositionLimit = VectorXd::Zero(model->nv);
    model->upperPositionLimit = VectorXd::Zero(model->nv);

    {
        SimulatorX sim = SimulatorX(model, geom_model);
        sim.settings.constraint_solvers_settings.admm_settings.absolute_feasibility_tol = 1e-9;
        sim.settings.constraint_solvers_settings.admm_settings.absolute_complementarity_tol = 1e-9;
        sim.settings.constraint_solvers_settings.admm_settings.relative_feasibility_tol = 1e-9;
        sim.settings.constraint_solvers_settings.admm_settings.relative_complementarity_tol = 1e-9;
        sim.step<ADMM>(q, v, tau, dt);
        EIGEN_VECTOR_IS_APPROX(sim.state.vnew, VectorXd::Zero(model->nv), 1e-6);
        // calling the simulator again should not change the state
        // and the solver should converge in one iteration
        sim.step<ADMM>(q, v, tau, dt);
        EIGEN_VECTOR_IS_APPROX(sim.state.vnew, VectorXd::Zero(model->nv), 1e-6);
        INDEX_EQUALITY_CHECK(sim.workspace.constraint_solvers.admm_result.iterations, 0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
