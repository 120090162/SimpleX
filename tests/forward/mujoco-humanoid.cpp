#include "simplex/core/simulator.hpp"
#include "simplex/core/simulator-x.hpp"
#include "simplex/utils/logger.hpp"

#include <pinocchio/algorithm/fwd.hpp>
#include <pinocchio/parsers/mjcf.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <memory>

#include "../test-utils.hpp"

using namespace simplex;
using namespace pinocchio;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(mujoco_humanoid_with_simulator)
{
    using ModelHandle = Simulator::ModelHandle;
    using GeometryModelHandle = Simulator::GeometryModelHandle;
#define ADMM ::pinocchio::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl

    ModelHandle model_handle(new Model());
    Model & model = ::pinocchio::helper::get_ref(model_handle);
    GeometryModelHandle geom_model_handle(new GeometryModel());
    GeometryModel & geom_model = ::pinocchio::helper::get_ref(geom_model_handle);

    const bool verbose = false;
    auto mjcf_path = findTestResource("SIMPLEX/tests/resources/mujoco_humanoid.xml");
    ::pinocchio::mjcf::buildModel(mjcf_path, model, verbose);
    ::pinocchio::mjcf::buildGeom(model, mjcf_path, pinocchio::COLLISION, geom_model);
    addFloor(geom_model);

    // sanity checks
    assert(model.nv == 27);
    assert(geom_model.geometryObjects.size() == 20);

    // initial state
    const Eigen::VectorXd q0 = model.referenceConfigurations["qpos0"];
    const Eigen::VectorXd v0 = Eigen::VectorXd::Zero(model.nv);
    const Eigen::VectorXd tau0 = Eigen::VectorXd::Zero(model.nv);

    // add collision pairs
    addCollisionPairs(model, geom_model, q0);
    assert(geom_model.collisionPairs.size() == 175);

    // run simulation
    model.lowerPositionLimit.setConstant(-std::numeric_limits<double>::infinity());
    model.upperPositionLimit.setConstant(std::numeric_limits<double>::infinity());
    model.lowerDryFrictionLimit.setZero();
    model.upperDryFrictionLimit.setZero();
    const double dt = 1e-3;
    const Eigen::VectorXd zero_torque = Eigen::VectorXd::Zero(model.nv);

    {
        Simulator sim(model_handle, geom_model_handle);
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        for (size_t i = 0; i < 100; ++i)
        {
            BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, zero_torque, dt));
            BOOST_CHECK(sim.admm_constraint_solver.getIterationCount() < sim.admm_constraint_solver_settings.max_iter);
            BOOST_CHECK(sim.pgs_constraint_solver.getIterationCount() == 0); // make sure pgs didnt run
            q = sim.qnew;
            v = sim.vnew;
        }

        std::cout << simplex::logging::DEBUG << "Simulator Final ADMM q" << q.transpose() << std::endl;
        std::cout << simplex::logging::DEBUG << "Simulator Final ADMM v" << v.transpose() << std::endl;
    }

    {
        Simulator sim(model_handle, geom_model_handle);
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        for (size_t i = 0; i < 100; ++i)
        {
            BOOST_CHECK_NO_THROW(sim.step<PGS>(q, v, zero_torque, dt));
            BOOST_CHECK(sim.pgs_constraint_solver.getIterationCount() < sim.pgs_constraint_solver_settings.max_iter);
            BOOST_CHECK(sim.admm_constraint_solver.getIterationCount() == 0); // make sure admm didnt run
            q = sim.qnew;
            v = sim.vnew;
        }

        std::cout << simplex::logging::DEBUG << "Simulator Final PGS q" << q.transpose() << std::endl;
        std::cout << simplex::logging::DEBUG << "Simulator Final PGS v" << v.transpose() << std::endl;
    }

#undef ADMM
#undef PGS
}

BOOST_AUTO_TEST_CASE(mujoco_humanoid_with_simulatorx)
{
    using ModelHandle = SimulatorX::ModelHandle;
    using GeometryModelHandle = SimulatorX::GeometryModelHandle;
#define ADMM ::simplex::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl
#define Clarabel ::simplex::ClarabelContactSolverTpl

    ModelHandle model_handle(new Model());
    Model & model = ::pinocchio::helper::get_ref(model_handle);
    GeometryModelHandle geom_model_handle(new GeometryModel());
    GeometryModel & geom_model = ::pinocchio::helper::get_ref(geom_model_handle);

    const bool verbose = false;
    const bool add_floor = true;

    load(model, geom_model, add_floor, verbose);

    // sanity checks
    assert(model.nv == 27);
    assert(geom_model.geometryObjects.size() == 20);

    // initial state
    const Eigen::VectorXd q0 = model.referenceConfigurations["qpos0"];
    const Eigen::VectorXd v0 = Eigen::VectorXd::Zero(model.nv);
    const Eigen::VectorXd tau0 = Eigen::VectorXd::Zero(model.nv);

    // add collision pairs
    addCollisionPairs(model, geom_model, q0);
    assert(geom_model.collisionPairs.size() == 175);

    // run simulation
    model.lowerPositionLimit.setConstant(-std::numeric_limits<double>::infinity());
    model.upperPositionLimit.setConstant(std::numeric_limits<double>::infinity());
    model.lowerDryFrictionLimit.setZero();
    model.upperDryFrictionLimit.setZero();
    const double dt = 1e-3;
    const Eigen::VectorXd zero_torque = Eigen::VectorXd::Zero(model.nv);

    {
        SimulatorX sim(model_handle, geom_model_handle);
        std::cout << simplex::logging::DEBUG
                  << "SimulatorX Max Contact Number: " << sim.workspace.constraint_problem().getMaxNumberOfContacts() << std::endl;
        std::cout << simplex::logging::DEBUG
                  << "SimulatorX Frictional Point Size: " << sim.workspace.constraint_problem().frictional_point_constraints_size()
                  << std::endl;
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        for (size_t i = 0; i < 100; ++i)
        {
            BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, zero_torque, dt));
            BOOST_CHECK(
                sim.workspace.constraint_solvers.admm_solver.getIterationCount()
                < sim.config.constraint_solvers_configs.admm_config.max_iter);
            BOOST_CHECK(sim.workspace.constraint_solvers.pgs_solver.getIterationCount() == 0);      // make sure pgs didnt run
            BOOST_CHECK(sim.workspace.constraint_solvers.clarabel_solver.getIterationCount() == 0); // make sure clarabel didnt run
            q = sim.state.qnew;
            v = sim.state.vnew;
        }

        std::cout << simplex::logging::DEBUG << "SimulatorX Final ADMM q" << q.transpose() << std::endl;
        std::cout << simplex::logging::DEBUG << "SimulatorX Final ADMM v" << v.transpose() << std::endl;
        std::cout << simplex::logging::DEBUG << "SimulatorX Contact Number: " << sim.workspace.constraint_problem().getNumberOfContacts()
                  << std::endl;
        std::cout << simplex::logging::DEBUG
                  << "SimulatorX Frictional Point Size: " << sim.workspace.constraint_problem().frictional_point_constraints_size()
                  << std::endl;
    }

    {
        SimulatorX sim(model_handle, geom_model_handle);
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        for (size_t i = 0; i < 100; ++i)
        {
            BOOST_CHECK_NO_THROW(sim.step<PGS>(q, v, zero_torque, dt));
            BOOST_CHECK(
                sim.workspace.constraint_solvers.pgs_solver.getIterationCount()
                < sim.config.constraint_solvers_configs.pgs_config.max_iter);
            BOOST_CHECK(sim.workspace.constraint_solvers.admm_solver.getIterationCount() == 0);     // make sure admm didnt run
            BOOST_CHECK(sim.workspace.constraint_solvers.clarabel_solver.getIterationCount() == 0); // make sure clarabel didnt run
            q = sim.state.qnew;
            v = sim.state.vnew;
        }

        std::cout << simplex::logging::DEBUG << "SimulatorX Final PGS q" << q.transpose() << std::endl;
        std::cout << simplex::logging::DEBUG << "SimulatorX Final PGS v" << v.transpose() << std::endl;
    }

    {
        SimulatorX sim(model_handle, geom_model_handle);
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        for (size_t i = 0; i < 100; ++i)
        {
            BOOST_CHECK_NO_THROW(sim.step<Clarabel>(q, v, zero_torque, dt));
            BOOST_CHECK(
                sim.workspace.constraint_solvers.clarabel_solver.getIterationCount()
                < sim.config.constraint_solvers_configs.clarabel_config.max_iter);
            BOOST_CHECK(sim.workspace.constraint_solvers.admm_solver.getIterationCount() == 0); // make sure admm didnt run
            BOOST_CHECK(sim.workspace.constraint_solvers.pgs_solver.getIterationCount() == 0);  // make sure pgs didnt run
            q = sim.state.qnew;
            v = sim.state.vnew;
        }

        std::cout << simplex::logging::DEBUG << "SimulatorX Final Clarabel q" << q.transpose() << std::endl;
        std::cout << simplex::logging::DEBUG << "SimulatorX Final Clarabel v" << v.transpose() << std::endl;
    }

#undef ADMM
#undef PGS
#undef Clarabel
}

BOOST_AUTO_TEST_SUITE_END()
