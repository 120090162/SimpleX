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
namespace pin = pinocchio;

using BilateralPointConstraintModel = SimulatorX::BilateralPointConstraintModel;
using BilateralPointConstraintModelVector = SimulatorX::BilateralPointConstraintModelVector;
using WeldConstraintModel = SimulatorX::WeldConstraintModel;
using WeldConstraintModelVector = SimulatorX::WeldConstraintModelVector;

using VectorXd = Eigen::VectorXd;
using MatrixXd = Eigen::MatrixXd;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(humanoid_with_point_anchor_constraint_minimal)
{
    // Construct humanoid
    pin::Model model;
    pin::buildModels::humanoid(model, true);
    pin::Data data(model);

    // Initial state
    VectorXd q = pin::neutral(model);
    VectorXd v = VectorXd::Zero(model.nv);
    VectorXd tau = VectorXd::Zero(model.nv);
    const double dt = 1e-3;

    // Compute vfree and crba for G and g.
    // Also populate data with the system's current state
    pin::crba(model, data, q, pin::Convention::WORLD);
    const VectorXd v_free = dt * pin::aba(model, data, q, v, tau, pin::Convention::WORLD);
    data.q_in = q;
    data.v_in = v;
    data.tau_in = tau;

    // Point anchor constraint on the robot right wrist
    pin::forwardKinematics(model, data, q);
    const pin::JointIndex joint1_id = 0;
    const pin::GeomIndex joint2_id = 14;
    assert((int)joint2_id < model.njoints);
    assert(model.nvs[joint2_id] == 1); // make sure its a bilaterable joint
    pin::SE3 Mc = data.oMi[joint2_id];
    const pin::SE3 joint1_placement = Mc;
    const pin::SE3 joint2_placement = pin::SE3::Identity();

    using ConstraintModel = BilateralPointConstraintModel;
    using ConstraintData = typename ConstraintModel::ConstraintData;
    std::vector<ConstraintModel> constraint_models;
    std::vector<ConstraintData> constraint_datas;
    ConstraintModel cm(model, joint1_id, joint1_placement, joint2_id, joint2_placement);
    constraint_models.push_back(cm);
    constraint_datas.push_back(cm.createData());
    for (std::size_t i = 0; i < constraint_models.size(); ++i)
    {
        constraint_models[i].calc(model, data, constraint_datas[i]);
    }
    const Eigen::DenseIndex constraint_size = pin::getTotalConstraintActiveSize(constraint_models);

    // Delassus
    pin::ContactCholeskyDecomposition chol(model, constraint_models);
    chol.compute(model, data, constraint_models, constraint_datas, 1e-10);

    // Solve constraint
    const MatrixXd delassus = chol.getDelassusCholeskyExpression().matrix();
    const pin::DelassusOperatorDense G(delassus);

    BOOST_CHECK(delassus.rows() == constraint_size);
    MatrixXd constraint_jacobian(delassus.rows(), model.nv);
    constraint_jacobian.setZero();
    getConstraintsJacobian(model, data, constraint_models, constraint_datas, constraint_jacobian);

    const VectorXd g = constraint_jacobian * v_free;

    pin::PGSContactSolver pgs_solver(std::size_t(delassus.rows()));
    pgs_solver.setAbsolutePrecision(1e-10);
    pgs_solver.setRelativePrecision(1e-14);
    pgs_solver.setMaxIterations(1000);
    BOOST_CHECK(constraint_size == delassus.rows());

    VectorXd primal_solution = VectorXd::Zero(constraint_size);
    const bool has_converged = pgs_solver.solve(
        G, g,                  //
        constraint_models, dt, //
        boost::make_optional((Eigen::Ref<const VectorXd>)primal_solution));
    BOOST_CHECK(has_converged);
    primal_solution = pgs_solver.getPrimalSolution();

    const VectorXd tau_ext = constraint_jacobian.transpose() * primal_solution / dt;
    VectorXd v_next = v + dt * pin::aba(model, data, q, v, (tau + tau_ext).eval(), pin::Convention::WORLD);
    VectorXd dv = v * dt;
    VectorXd q_next = pin::integrate(model, q, dv);

    VectorXd v_wrist_expected = VectorXd::Zero(model.nvs[joint2_id]);

    EIGEN_VECTOR_IS_APPROX(
        v_next.segment(model.idx_vs[joint2_id], model.nvs[joint2_id]), //
        v_wrist_expected,                                              //
        1e-6);
    pin::forwardKinematics(model, data, q_next);
    EIGEN_VECTOR_IS_APPROX(Mc.translation(), data.oMi[joint2_id].translation(), 1e-6);
}

BOOST_AUTO_TEST_CASE(humanoid_with_frame_anchor_constraint_minimal)
{
    // Construct humanoid
    pin::Model model;
    pin::buildModels::humanoid(model, true);
    pin::Data data(model);

    // Initial state
    VectorXd q = pin::neutral(model);
    VectorXd v = VectorXd::Zero(model.nv);
    VectorXd tau = VectorXd::Zero(model.nv);
    const double dt = 1e-3;

    // Compute vfree and crba for G and g
    // Populate data with system current state
    pin::crba(model, data, q, pin::Convention::WORLD);
    const VectorXd v_free = dt * pin::aba(model, data, q, v, tau, pin::Convention::WORLD);
    data.q_in = q;
    data.v_in = v;
    data.tau_in = tau;

    // Point anchor constraint on the robot right wrist
    pin::forwardKinematics(model, data, q);
    const pin::JointIndex joint1_id = 0;
    const pin::GeomIndex joint2_id = 14;
    assert((int)joint2_id < model.njoints);
    assert(model.nvs[joint2_id] == 1); // make sure its a bilaterable joint
    pin::SE3 Mc = data.oMi[joint2_id];
    const pin::SE3 joint1_placement = Mc;
    const pin::SE3 joint2_placement = pin::SE3::Identity();

    using ConstraintModel = BilateralPointConstraintModel;
    using ConstraintData = typename ConstraintModel::ConstraintData;
    std::vector<ConstraintModel> constraint_models;
    std::vector<ConstraintData> constraint_datas;
    ConstraintModel cm(model, joint1_id, joint1_placement, joint2_id, joint2_placement);
    constraint_models.push_back(cm);
    constraint_datas.push_back(cm.createData());

    for (std::size_t i = 0; i < constraint_models.size(); ++i)
    {
        constraint_models[i].calc(model, data, constraint_datas[i]);
    }

    const Eigen::DenseIndex constraint_size = pin::getTotalConstraintActiveSize(constraint_models);

    // Delassus
    pin::ContactCholeskyDecomposition chol(model, constraint_models);
    chol.compute(model, data, constraint_models, constraint_datas, 1e-10);

    // Solve constraint
    const MatrixXd delassus = chol.getDelassusCholeskyExpression().matrix();
    const pin::DelassusOperatorDense G(delassus);

    MatrixXd constraint_jacobian(delassus.rows(), model.nv);
    constraint_jacobian.setZero();
    getConstraintsJacobian(model, data, constraint_models, constraint_datas, constraint_jacobian);

    const VectorXd g = constraint_jacobian * v_free;

    pin::PGSContactSolver pgs_solver(std::size_t(delassus.rows()));
    pgs_solver.setAbsolutePrecision(1e-10);
    pgs_solver.setRelativePrecision(1e-14);
    pgs_solver.setMaxIterations(1000);
    BOOST_CHECK(constraint_size == delassus.rows());

    VectorXd primal_solution = VectorXd::Zero(constraint_size);

    const bool has_converged = pgs_solver.solve(
        G, g,                  //
        constraint_models, dt, //
        boost::make_optional((Eigen::Ref<const VectorXd>)primal_solution));
    BOOST_CHECK(has_converged);
    primal_solution = pgs_solver.getPrimalSolution();

    const VectorXd tau_ext = constraint_jacobian.transpose() * primal_solution / dt;
    VectorXd v_next = v + dt * pin::aba(model, data, q, v, (tau + tau_ext).eval(), pin::Convention::WORLD);
    VectorXd dv = v * dt;
    VectorXd q_next = pin::integrate(model, q, dv);

    VectorXd v_wrist_expected = VectorXd::Zero(model.nvs[joint2_id]);

    EIGEN_VECTOR_IS_APPROX(
        v_next.segment(model.idx_vs[joint2_id], model.nvs[joint2_id]), //
        v_wrist_expected,                                              //
        1e-6);
    pin::forwardKinematics(model, data, q_next);
    EIGEN_VECTOR_IS_APPROX(Mc.translation(), data.oMi[joint2_id].translation(), 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
