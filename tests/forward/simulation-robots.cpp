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

BOOST_AUTO_TEST_CASE(humanoid_with_bilateral_constraint)
{
    // Construct humanoid
    ModelHandle model(new Model());
    buildModels::humanoid(*model, true);
    model->lowerPositionLimit = Eigen::VectorXd::Constant(model->nq, -1.0) * std::numeric_limits<double>::max();
    model->upperPositionLimit = Eigen::VectorXd::Constant(model->nq, 1.0) * std::numeric_limits<double>::max();
    DataHandle data = std::make_shared<Data>(*model);

    GeometryModelHandle geom_model(new GeometryModel());

    // Initial state
    Eigen::VectorXd q = pinocchio::neutral(*model);

    // Bilateral constraint on the robot right wrist
    BilateralPointConstraintModelVector bilateral_constraint_models;
    pinocchio::framesForwardKinematics(*model, *data, q);
    const JointIndex joint1_id = 0;
    const GeomIndex joint2_id = 13;
    ::pinocchio::SE3 Mc = data->oMi[joint2_id];
    const SE3 joint1_placement = Mc;
    const SE3 joint2_placement = SE3::Identity();
    bilateral_constraint_models.push_back(BilateralPointConstraintModel(*model, joint1_id, joint1_placement, joint2_id, joint2_placement));
    bilateral_constraint_models[0].baumgarte_corrector_parameters().Kp = 0.1;

    // The humanoid's freeflyer's height should remain the same
    Simulator sim(model, geom_model, bilateral_constraint_models);
    sim.admm_constraint_solver_settings.absolute_precision = 1e-9;
    sim.admm_constraint_solver_settings.relative_precision = 1e-9;
    sim.admm_constraint_solver_settings.max_iter = 1000;
    const double dt = 1e-3;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
    pinocchio::framesForwardKinematics(*model, *data, sim.qnew);
    EIGEN_VECTOR_IS_APPROX(Mc.translation(), data->oMi[joint2_id].translation(), 1e-3);

    // Calling the simulator twice to test warmstart
    sim.step<ADMM>(q, v, tau, dt);
    INDEX_EQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 0);

    for (int i = 0; i < 10; ++i)
    {
        Eigen::VectorXd q = sim.qnew;
        Eigen::VectorXd v = sim.vnew;
        BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
        pinocchio::framesForwardKinematics(*model, *data, sim.qnew);
        EIGEN_VECTOR_IS_APPROX(Mc.translation(), data->oMi[joint2_id].translation(), 1e-3);
    }
}

BOOST_AUTO_TEST_CASE(simulator_instance_step_with_friction_on_joints)
{
    ModelHandle model(new Model());
    buildModels::manipulator(*model);
    model->lowerDryFrictionLimit = Eigen::VectorXd::Ones(model->nv) * -100000.0;
    model->upperDryFrictionLimit = Eigen::VectorXd::Ones(model->nv) * 100000.0;

    GeometryModelHandle geom_model(new GeometryModel());
    buildModels::manipulatorGeometries(*model, *geom_model);

    Simulator sim(model, geom_model);

    Eigen::VectorXd q = neutral(*model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
    // calling the simulator again should not change the state
    // and the solver should converge in one iteration
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, v, 1e-6);
    EIGEN_VECTOR_IS_APPROX(sim.qnew, q, 1e-6);
    INDEX_EQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 0);
}

BOOST_AUTO_TEST_CASE(simulator_instance_step_with_limits_on_joints_for_manipulator)
{
    ModelHandle model(new Model());
    buildModels::manipulator(*model);
    // We first consider not active limits
    model->lowerPositionLimit = Eigen::VectorXd::Ones(model->nq) * -100000.0;
    model->upperPositionLimit = Eigen::VectorXd::Ones(model->nq) * 100000.0;

    GeometryModelHandle geom_model(new GeometryModel());
    buildModels::manipulatorGeometries(*model, *geom_model);

    Simulator sim(model, geom_model);

    Eigen::VectorXd q = neutral(*model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);
    // simulated velocity should be the same as the one computed with aba as constraints are not active
    Eigen::VectorXd vnew = v + dt * pinocchio::aba(*model, sim.data(), q, v, tau);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, vnew, 1e-6);
    // calling the simulator again should not change the state
    // and the solver should converge in one iteration
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, vnew, 1e-6);
    INDEX_EQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 0);

    // we now consider active limits
    q = Eigen::VectorXd::Zero(model->nv);
    model->lowerPositionLimit = Eigen::VectorXd::Zero(model->nv);
    model->upperPositionLimit = Eigen::VectorXd::Zero(model->nv);

    sim = Simulator(model, geom_model);
    sim.admm_constraint_solver_settings.absolute_precision = 1e-9;
    sim.admm_constraint_solver_settings.relative_precision = 1e-9;
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
    // calling the simulator again should not change the state
    // and the solver should converge in one iteration
    sim.step<ADMM>(q, v, tau, dt);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
    INDEX_EQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 0);
}

BOOST_AUTO_TEST_SUITE_END()
