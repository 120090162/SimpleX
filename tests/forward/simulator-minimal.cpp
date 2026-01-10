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
using BilateralPointConstraintModel = Simulator::BilateralPointConstraintModel;
using BilateralPointConstraintModelVector = Simulator::BilateralPointConstraintModelVector;
using WeldConstraintModel = Simulator::WeldConstraintModel;
using WeldConstraintModelVector = Simulator::WeldConstraintModelVector;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(humanoid_with_bilateral_constraint_minimal)
{
    // Construct humanoid
    Model model;
    buildModels::humanoid(model, true);
    Data data(model);

    // Initial state
    Eigen::VectorXd q = pinocchio::neutral(model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model.nv);
    const double dt = 1e-3;

    // Compute vfree and crba for G and g
    crba(model, data, q, Convention::WORLD);
    const Eigen::VectorXd v_free = dt * aba(model, data, q, v, tau, Convention::WORLD);

    // Bilateral constraint on the robot right wrist
    pinocchio::forwardKinematics(model, data, q);
    const JointIndex joint1_id = 0;
    const GeomIndex joint2_id = 14;
    assert((int)joint2_id < model.njoints);
    assert(model.nvs[joint2_id] == 1); // make sure its a bilaterable joint
    ::pinocchio::SE3 Mc = data.oMi[joint2_id];
    const SE3 joint1_placement = Mc;
    const SE3 joint2_placement = SE3::Identity();

    using ConstraintModel = BilateralPointConstraintModel;
    using ConstraintData = typename ConstraintModel::ConstraintData;
    std::vector<ConstraintModel> constraint_models;
    std::vector<ConstraintData> constraint_datas;
    ConstraintModel cm(model, joint1_id, joint1_placement, joint2_id, joint2_placement);
    constraint_models.push_back(cm);
    constraint_datas.push_back(cm.createData());
    const Eigen::DenseIndex constraint_size = getTotalConstraintSize(constraint_models);

    // Delassus
    ContactCholeskyDecomposition chol(model, constraint_models);
    chol.compute(model, data, constraint_models, constraint_datas, 1e-10);

    // Solve constraint
    const Eigen::MatrixXd delassus = chol.getDelassusCholeskyExpression().matrix();
    const DelassusOperatorDense G(delassus);

    Eigen::MatrixXd constraint_jacobian(delassus.rows(), model.nv);
    constraint_jacobian.setZero();
    getConstraintsJacobian(model, data, constraint_models, constraint_datas, constraint_jacobian);

    const Eigen::VectorXd g = constraint_jacobian * v_free;

    PGSContactSolver pgs_solver(int(delassus.rows()));
    pgs_solver.setAbsolutePrecision(1e-10);
    pgs_solver.setRelativePrecision(1e-14);
    pgs_solver.setMaxIterations(1000);
    Eigen::VectorXd primal_solution = Eigen::VectorXd::Zero(constraint_size);
    const bool has_converged =
        pgs_solver.solve(G, g, constraint_models, dt, boost::make_optional((Eigen::Ref<const Eigen::VectorXd>)primal_solution));
    BOOST_CHECK(has_converged);
    primal_solution = pgs_solver.getPrimalSolution();

    const Eigen::VectorXd tau_ext = constraint_jacobian.transpose() * primal_solution / dt;
    Eigen::VectorXd v_next = v + dt * aba(model, data, q, v, (tau + tau_ext).eval(), Convention::WORLD);
    Eigen::VectorXd dv = v * dt;
    Eigen::VectorXd q_next = ::pinocchio::integrate(model, q, dv);

    Eigen::VectorXd v_wrist_expected = Eigen::VectorXd::Zero(model.nvs[joint2_id]);

    EIGEN_VECTOR_IS_APPROX(
        v_next.segment(model.idx_vs[joint2_id], model.nvs[joint2_id]), //
        v_wrist_expected,                                              //
        1e-6);
    pinocchio::forwardKinematics(model, data, q_next);
    EIGEN_VECTOR_IS_APPROX(Mc.translation(), data.oMi[joint2_id].translation(), 1e-6);
}

BOOST_AUTO_TEST_CASE(humanoid_with_weld_constraint_minimal)
{
    // Construct humanoid
    Model model;
    buildModels::humanoid(model, true);
    Data data(model);

    // Initial state
    Eigen::VectorXd q = pinocchio::neutral(model);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model.nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model.nv);
    const double dt = 1e-3;

    // Compute vfree and crba for G and g
    crba(model, data, q, Convention::WORLD);
    const Eigen::VectorXd v_free = dt * aba(model, data, q, v, tau, Convention::WORLD);

    // Bilateral constraint on the robot right wrist
    pinocchio::forwardKinematics(model, data, q);
    const JointIndex joint1_id = 0;
    const GeomIndex joint2_id = 14;
    assert((int)joint2_id < model.njoints);
    assert(model.nvs[joint2_id] == 1); // make sure its a bilaterable joint
    ::pinocchio::SE3 Mc = data.oMi[joint2_id];
    const SE3 joint1_placement = Mc;
    const SE3 joint2_placement = SE3::Identity();

    using ConstraintModel = BilateralPointConstraintModel;
    using ConstraintData = typename ConstraintModel::ConstraintData;
    std::vector<ConstraintModel> constraint_models;
    std::vector<ConstraintData> constraint_datas;
    ConstraintModel cm(model, joint1_id, joint1_placement, joint2_id, joint2_placement);
    constraint_models.push_back(cm);
    constraint_datas.push_back(cm.createData());
    const Eigen::DenseIndex constraint_size = getTotalConstraintSize(constraint_models);

    // Delassus
    ContactCholeskyDecomposition chol(model, constraint_models);
    chol.compute(model, data, constraint_models, constraint_datas, 1e-10);

    // Solve constraint
    const Eigen::MatrixXd delassus = chol.getDelassusCholeskyExpression().matrix();
    const DelassusOperatorDense G(delassus);

    Eigen::MatrixXd constraint_jacobian(delassus.rows(), model.nv);
    constraint_jacobian.setZero();
    getConstraintsJacobian(model, data, constraint_models, constraint_datas, constraint_jacobian);

    const Eigen::VectorXd g = constraint_jacobian * v_free;

    PGSContactSolver pgs_solver(int(delassus.rows()));
    pgs_solver.setAbsolutePrecision(1e-10);
    pgs_solver.setRelativePrecision(1e-14);
    pgs_solver.setMaxIterations(1000);
    Eigen::VectorXd primal_solution = Eigen::VectorXd::Zero(constraint_size);
    const bool has_converged =
        pgs_solver.solve(G, g, constraint_models, dt, boost::make_optional((Eigen::Ref<const Eigen::VectorXd>)primal_solution));
    BOOST_CHECK(has_converged);
    primal_solution = pgs_solver.getPrimalSolution();

    const Eigen::VectorXd tau_ext = constraint_jacobian.transpose() * primal_solution / dt;
    Eigen::VectorXd v_next = v + dt * aba(model, data, q, v, (tau + tau_ext).eval(), Convention::WORLD);
    Eigen::VectorXd dv = v * dt;
    Eigen::VectorXd q_next = ::pinocchio::integrate(model, q, dv);

    Eigen::VectorXd v_wrist_expected = Eigen::VectorXd::Zero(model.nvs[joint2_id]);

    EIGEN_VECTOR_IS_APPROX(
        v_next.segment(model.idx_vs[joint2_id], model.nvs[joint2_id]), //
        v_wrist_expected,                                              //
        1e-6);
    pinocchio::forwardKinematics(model, data, q_next);
    EIGEN_VECTOR_IS_APPROX(Mc.translation(), data.oMi[joint2_id].translation(), 1e-6);
}

BOOST_AUTO_TEST_SUITE_END()
