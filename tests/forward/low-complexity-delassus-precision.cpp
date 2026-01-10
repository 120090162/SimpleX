#include "simplex/core/simulator-x.hpp"

#include <pinocchio/algorithm/fwd.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <memory>

#include "../test-utils.hpp"

using namespace simplex;
namespace pin = pinocchio;
using ModelHandle = SimulatorX::ModelHandle;
using DataHandle = SimulatorX::DataHandle;
using GeometryModelHandle = SimulatorX::GeometryModelHandle;
using GeometryDataHandle = SimulatorX::GeometryDataHandle;
#define ADMM ::simplex::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl
#define Clarabel ::simplex::ClarabelContactSolverTpl

using VectorXd = Eigen::VectorXd;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(mujoco_humanoid)
{
    ModelHandle model_handle(new pin::Model());
    pin::Model & model = ::pinocchio::helper::get_ref(model_handle);
    GeometryModelHandle geom_model_handle(new pin::GeometryModel());
    pin::GeometryModel & geom_model = ::pinocchio::helper::get_ref(geom_model_handle);

    const bool verbose = false;
    const bool add_floor = true;

    load(model, geom_model, add_floor, verbose);

    // sanity checks
    assert(model.nv == 27);
    assert(geom_model.geometryObjects.size() == 20);

    // initial state
    const VectorXd q0 = model.referenceConfigurations["qpos0"];
    const VectorXd v0 = VectorXd::Zero(model.nv);
    const VectorXd tau0 = VectorXd::Zero(model.nv);

    // add collision pairs
    addCollisionPairs(model, geom_model, q0);
    assert(geom_model.collisionPairs.size() == 175);

    // run simulation
    model.lowerPositionLimit.setConstant(-std::numeric_limits<double>::infinity());
    model.upperPositionLimit.setConstant(std::numeric_limits<double>::infinity());
    model.lowerDryFrictionLimit.setZero();
    model.upperDryFrictionLimit.setZero();
    const double dt = 1e-3;
    const VectorXd zero_torque = VectorXd::Zero(model.nv);

    const size_t debug_iter = 3;
    {
        SimulatorX sim_cholesky(model_handle, geom_model_handle);
        sim_cholesky.workspace.constraint_problem().delassus_type = SimulatorX::ConstraintsProblemDerivatives::DelassusType::CHOLESKY;

        VectorXd q = q0;
        VectorXd v = v0;
        for (size_t i = 0; i <= debug_iter; ++i)
        {
            BOOST_CHECK_NO_THROW(sim_cholesky.step<ADMM>(q, v, zero_torque, dt));
            if (i < debug_iter)
            {
                q = sim_cholesky.state.qnew;
                v = sim_cholesky.state.vnew;
            }
        }

        using DelassusSparseMultibody = SimulatorX::DelassusCholeskyExpressionOperator;
        DelassusSparseMultibody delassus_sparse_multibody(sim_cholesky.workspace.constraint_problem().constraint_cholesky_decomposition);
        const auto dense_sparse_multibody = delassus_sparse_multibody.matrix();

        // Create a new Delassus low complexity
        // Note: because `step` was called, data/constraint_datas are up to date with the system's state
        using DelassusLowComplexity = SimulatorX::DelassusRigidBodyOperator;
        DelassusLowComplexity delassus_low_complexity(
            pinocchio::helper::make_ref(sim_cholesky.model()),                                          //
            pinocchio::helper::make_ref(sim_cholesky.data()),                                           //
            pinocchio::helper::make_ref(sim_cholesky.workspace.constraint_problem().constraint_models), //
            pinocchio::helper::make_ref(sim_cholesky.workspace.constraint_problem().constraint_datas));

        delassus_low_complexity.updateDamping(delassus_sparse_multibody.getDamping());
        delassus_low_complexity.updateCompliance(delassus_sparse_multibody.getCompliance());
        delassus_low_complexity.compute(true, true);
        const auto dense_low_complexity = delassus_low_complexity.matrix();

        std::cout << "Delassus size: (" << delassus_low_complexity.rows() << ", " << delassus_low_complexity.cols() << ")" << std::endl;
        std::cout << "dense_sparse_multibody:\n" << dense_sparse_multibody << std::endl;
        std::cout << "dense_low_complexity:\n" << dense_low_complexity << std::endl;

        const VectorXd vector_ones = VectorXd::Ones(delassus_low_complexity.rows());
        VectorXd res_apply_on_the_right = VectorXd::Zero(delassus_low_complexity.rows());
    }
}

BOOST_AUTO_TEST_SUITE_END()
