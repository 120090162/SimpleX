#include "simplex/core/simulator-x.hpp"
#include "simplex/core/simulator-derivatives.hpp"

#include <pinocchio/algorithm/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <memory>

#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include <fkYAML/node.hpp>

#include "../test-utils.hpp"

using namespace simplex;
using namespace pinocchio;
using ModelHandle = SimulatorX::ModelHandle;
using DataHandle = SimulatorX::DataHandle;
using GeometryModelHandle = SimulatorX::GeometryModelHandle;
using GeometryDataHandle = SimulatorX::GeometryDataHandle;

using json = nlohmann::json;

#define ADMM ::simplex::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl
#define Clarabel ::simplex::ClarabelContactSolverTpl

std::string getContactSolverType()
{
    static std::string type;
    if (type.empty())
    {
        fkyaml::node j;
        // json j;
        try
        {
            std::ifstream jfile(findTestResource("SIMPLEX/tests/config/derivatives-simulator.yaml"));
            // std::ifstream jfile(findTestResource("SIMPLEX/tests/config/derivatives-simulator.json"));
            if (jfile.good())
            {
                jfile >> j;
                if (j.contains("ContactSolver") && j["ContactSolver"].contains("type"))
                {
                    type = j["ContactSolver"]["type"].get_value<std::string>();
                    // type = j["ContactSolver"]["type"].get<std::string>();
                }
                else
                {
                    type = "admm";
                }
            }
            else
            {
                type = "admm";
            }
        }
        catch (...)
        {
            type = "admm";
        }
    }
    return type;
}

void step(SimulatorX & sim, const Eigen::VectorXd & q, const Eigen::VectorXd & v, const Eigen::VectorXd & tau, const double dt)
{
    std::string type = getContactSolverType();
    if (type == "pgs")
    {
        sim.step<PGS>(q, v, tau, dt);
    }
    else if (type == "admm")
    {
        sim.step<ADMM>(q, v, tau, dt);
    }
    else if (type == "clarabel")
    {
        sim.step<Clarabel>(q, v, tau, dt);
    }
    else
    {
        throw std::runtime_error("Unknown solver type: " + type);
    }
}
namespace ns
{
    struct novel
    {
        std::string title;
        std::string author;
        int year;
    };

    struct recommend
    {
        std::string title;
        std::string author;
    };

    // overloads must be defined in the same namespace as user-defined types.
    void from_node(const fkyaml::node & node, novel & novel)
    {
        novel.title = node["title"].as_str();
        novel.author = node["author"].as_str();
        novel.year = node["year"].get_value<int>();
    }

    void to_node(fkyaml::node & node, const recommend & recommend)
    {
        node = fkyaml::node{{"title", recommend.title}, {"author", recommend.author}};
    }

} // namespace ns

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(simulator_derivatives_read_yaml_config)
{
    // open a YAML file. Other streams or strings are also usable as an input.
    std::ifstream ifs(findTestResource("SIMPLEX/tests/config/test.yaml"));
    // deserialize the loaded file contents.
    fkyaml::node root = fkyaml::node::deserialize(ifs);

    // print the deserialized YAML nodes by serializing them back.
    std::cout << root << std::endl;

    BOOST_CHECK(root.contains("novels"));
    BOOST_CHECK(root.contains("pi"));
    BOOST_CHECK(root.contains("happy"));
    BOOST_CHECK_EQUAL(root["pi"].get_value<float>(), float(3.1415));
    BOOST_CHECK_EQUAL(root["happy"].get_value<bool>(), true);

    // get novels directly from the node.
    auto novels = root["novels"].get_value<std::vector<ns::novel>>();

    for (auto & novel : novels)
    {
        std::cout << "Title: " << novel.title << ", Author: " << novel.author << ", Year: " << novel.year << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_read_json_config)
{
    json j;
    std::ifstream jfile(findTestResource("SIMPLEX/tests/config/test.json"));
    jfile >> j;
    BOOST_CHECK(j.contains("pi"));
    BOOST_CHECK(j.contains("happy"));
    BOOST_CHECK_EQUAL(j["pi"].get<float>(), float(3.1415));
    BOOST_CHECK_EQUAL(j["happy"].get<bool>(), true);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_constructor)
{
    ModelHandle model(new Model());
    GeometryModelHandle geom_model(new GeometryModel());
    SimulatorX sim(model, geom_model);
    BOOST_CHECK_NO_THROW(SimulatorDerivatives dsim(sim));
}

void computeStepDerivativesFD(
    SimulatorX & sim_fd,
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const Eigen::VectorXd & tau,
    const double dt,
    Eigen::MatrixXd & dvnew_dq_fd,
    Eigen::MatrixXd & dvnew_dv_fd,
    Eigen::MatrixXd & dvnew_dtau_fd)
{
    step(sim_fd, q, v, tau, dt);
    // sim_fd.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd vnew = sim_fd.state.vnew;
    double delta = 1e-6;
    // finite differences on q
    for (int i = 0; i < sim_fd.model().nv; i++)
    {
        Eigen::Index colid = (Eigen::Index)i;
        Eigen::VectorXd qplus = q;
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(sim_fd.model().nv);
        dq(colid) = delta;
        qplus = pinocchio::integrate(sim_fd.model(), q, dq);
        step(sim_fd, qplus, v, tau, dt);
        // sim_fd.step<ADMM>(qplus, v, tau, dt);
        dvnew_dq_fd.col(colid) = (sim_fd.state.vnew - vnew) / delta;
    }
    // finite differences on v
    for (int i = 0; i < sim_fd.model().nv; i++)
    {
        Eigen::Index colid = (Eigen::Index)i;
        Eigen::VectorXd vplus = v;
        Eigen::VectorXd dv = Eigen::VectorXd::Zero(sim_fd.model().nv);
        dv(colid) = delta;
        vplus = v + dv;
        step(sim_fd, q, vplus, tau, dt);
        // sim_fd.step<ADMM>(q, vplus, tau, dt);
        dvnew_dv_fd.col(colid) = (sim_fd.state.vnew - vnew) / delta;

        // sim_fd.workspace.workspace.constraint_problem().collectActiveSet();
        // for (std::size_t i = 0; i < sim_fd.workspace.workspace.constraint_problem().getContactModes().size(); ++i)
        // {
        //   ConstraintProblem::ContactMode cat = sim_fd.workspace.workspace.constraint_problem().getContactMode(i);
        //   switch (cat) {
        //   case ConstraintProblem::ContactMode::BREAKING:
        //       std::cout << "contact mode " << i << "  BREAKING" << std::endl;
        //       break;
        //   case ConstraintProblem::ContactMode::SLIDING:
        //       std::cout << "contact mode " << i << "  SLIDING" << std::endl;
        //       break;
        //   case ConstraintProblem::ContactMode::STICKING:
        //       std::cout << "contact mode " << i << "  STICKING" << std::endl;
        //   }
        // }
    }
    // finite differences on tau
    for (int i = 0; i < sim_fd.model().nv; i++)
    {
        Eigen::Index colid = (Eigen::Index)i;
        Eigen::VectorXd tauplus = tau;
        Eigen::VectorXd dtau = Eigen::VectorXd::Zero(sim_fd.model().nv);
        dtau(colid) = delta;
        tauplus = tau + dtau;
        step(sim_fd, q, v, tauplus, dt);
        // sim_fd.step<ADMM>(q, v, tauplus, dt);
        dvnew_dtau_fd.col(colid) = (sim_fd.state.vnew - vnew) / delta;
    }
}

void computeLambdaDerivativesFD(
    SimulatorX & sim_fd,
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & v,
    const Eigen::VectorXd & tau,
    const double dt,
    Eigen::MatrixXd & dlam_dq_fd,
    Eigen::MatrixXd & dlam_dv_fd,
    Eigen::MatrixXd & dlam_dtau_fd)
{
    // This function computes the variations of the contact forces with respect to q, v and tau using
    // finite differences. The implementation is very naive (even wrong in general) as it assumes that
    // the contact points remain the same
    sim_fd.reset();
    step(sim_fd, q, v, tau, dt);
    // sim_fd.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd lam = sim_fd.workspace.constraint_problem().frictional_point_constraints_forces();
    double delta = 1e-6;
    // finite differences on q
    sim_fd.reset();
    for (int i = 0; i < sim_fd.model().nv; i++)
    {
        Eigen::Index colid = (Eigen::Index)i;
        Eigen::VectorXd qplus = q;
        Eigen::VectorXd dq = Eigen::VectorXd::Zero(sim_fd.model().nv);
        dq(colid) = delta;
        qplus = pinocchio::integrate(sim_fd.model(), q, dq);
        step(sim_fd, qplus, v, tau, dt);
        // sim_fd.step<ADMM>(qplus, v, tau, dt);
        dlam_dq_fd.col(colid) = (sim_fd.workspace.constraint_problem().frictional_point_constraints_forces() - lam) / delta;
        sim_fd.reset();
    }
    // finite differences on v
    for (int i = 0; i < sim_fd.model().nv; i++)
    {
        Eigen::Index colid = (Eigen::Index)i;
        Eigen::VectorXd vplus = v;
        Eigen::VectorXd dv = Eigen::VectorXd::Zero(sim_fd.model().nv);
        dv(colid) = delta;
        vplus = v + dv;
        step(sim_fd, q, vplus, tau, dt);
        // sim_fd.step<ADMM>(q, vplus, tau, dt);
        dlam_dv_fd.col(colid) = (sim_fd.workspace.constraint_problem().frictional_point_constraints_forces() - lam) / delta;

        // sim_fd.workspace.constraint_problem().collectActiveSet();
        // for (std::size_t i = 0; i < sim_fd.workspace.constraint_problem().getContactModes().size(); ++i)
        // {
        //   ConstraintProblem::ContactMode cat = sim_fd.workspace.constraint_problem().getContactMode(i);
        //   switch (cat) {
        //   case ConstraintProblem::ContactMode::BREAKING:
        //       std::cout << "contact mode " << i << "  BREAKING" << std::endl;
        //       break;
        //   case ConstraintProblem::ContactMode::SLIDING:
        //       std::cout << "contact mode " << i << "  SLIDING" << std::endl;
        //       break;
        //   case ConstraintProblem::ContactMode::STICKING:
        //       std::cout << "contact mode " << i << "  STICKING" << std::endl;
        //   }
        // }
        sim_fd.reset();
    }
    // finite differences on tau
    for (int i = 0; i < sim_fd.model().nv; i++)
    {
        Eigen::Index colid = (Eigen::Index)i;
        Eigen::VectorXd tauplus = tau;
        Eigen::VectorXd dtau = Eigen::VectorXd::Zero(sim_fd.model().nv);
        dtau(colid) = delta;
        tauplus = tau + dtau;
        step(sim_fd, q, v, tauplus, dt);
        // sim_fd.step<ADMM>(q, v, tauplus, dt);
        dlam_dtau_fd.col(colid) = (sim_fd.workspace.constraint_problem().frictional_point_constraints_forces() - lam) / delta;
        sim_fd.reset();
    }
}

#ifdef PINOCCHIO_WITH_HPP_FCL
BOOST_AUTO_TEST_CASE(simulator_derivatives_ball_plane)
{
    // This case tests derivatives with a single (both sliding and sticking) contact between a ball
    // and a plane.
    Model _model;
    GeometryModel _geom_model;
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const FrameIndex plane_frame = FrameIndex(0);
    const GeometryObject plane_geom = GeometryObject("plane", 0, plane_frame, SE3::Identity(), plane_ptr);
    GeomIndex plane_id = _geom_model.addGeometryObject(plane_geom);

    const std::string name = "ball_1";
    const std::string joint_name = "joint_1";
    const std::string frame_name = "frame_1";
    const double mass = 1.0;
    const double r = 1.0;
    const Inertia inertia = Inertia::FromSphere(mass, r);
    const JointModelFreeFlyer joint = JointModelFreeFlyer();
    const JointIndex parent = 0;
    const SE3 placement = SE3::Identity();
    JointIndex joint_id = _model.addJoint(parent, joint, placement, joint_name);
    _model.appendBodyToJoint(joint_id, inertia);
    coal::CollisionGeometryPtr_t sphere_ptr(new coal::Sphere(r));
    const GeometryObject sphere_geom = GeometryObject(name, joint_id, placement, sphere_ptr);
    GeomIndex box_id = _geom_model.addGeometryObject(sphere_geom);
    CollisionPair cp(plane_id, box_id);
    _geom_model.addCollisionPair(cp);

    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-15;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-15;

    // Testing sticking mode
    Eigen::VectorXd q = neutral(*model);
    q(2) = (r / 2.) * .8;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd qnew = sim.state.qnew;
    Eigen::VectorXd vnew = sim.state.vnew;

    SimulatorDerivatives dsim(sim);
    dsim.stepDerivatives(sim, q, v, tau, dt);
    Eigen::MatrixXd dvnew_dq = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq = dsim.contact_solver_derivatives.dlam_dtheta().leftCols(sim.model().nv);
    Eigen::MatrixXd dlam_dv = dsim.contact_solver_derivatives.dlam_dtheta().middleCols(sim.model().nv, sim.model().nv);
    Eigen::MatrixXd dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);

    Eigen::MatrixXd dvnew_dq_fd = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv_fd = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau_fd = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq_fd = dlam_dq;
    Eigen::MatrixXd dlam_dv_fd = dlam_dv;
    Eigen::MatrixXd dlam_dtau_fd = dlam_dtau;

    SimulatorX sim_fd(model, geom_model);
    sim_fd.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim_fd.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-15;
    sim_fd.config.constraint_solvers_configs.admm_config.relative_precision = 1e-15;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    // std::cout << "q " << q << std::endl;
    // std::cout << "v " << v << std::endl;
    // std::cout << "qnew   " << qnew << std::endl;
    // std::cout << "vnew   " << vnew << std::endl;
    // std::cout << "contact forces" << sim.workspace.constraint_problem().point_contact_constraint_forces();
    // for (int i = 0; i < sim.workspace.constraint_problem().getContactModes().size(); ++i)
    // {
    //   ConstraintProblem::ContactMode cat = sim.workspace.constraint_problem().getContactMode(i);
    //   switch (cat) {
    //   case ConstraintProblem::ContactMode::BREAKING:
    //       std::cout << "contact mode " << i << "  BREAKING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::SLIDING:
    //       std::cout << "contact mode " << i << "  SLIDING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::STICKING:
    //       std::cout << "contact mode " << i << "  STICKING" << std::endl;
    //   }
    // }
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 5e-3);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 1e-3);

    // testing sliding mode
    v(0) = 1;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    qnew = sim.state.qnew;
    vnew = sim.state.vnew;

    dsim.stepDerivatives(sim, q, v, tau, dt);
    dvnew_dq = dsim.dvnew_dq;
    dvnew_dv = dsim.dvnew_dv;
    dvnew_dtau = dsim.dvnew_dtau;
    dlam_dq = dsim.contact_solver_derivatives.dlam_dtheta().leftCols(sim.model().nv);
    dlam_dv = dsim.contact_solver_derivatives.dlam_dtheta().middleCols(sim.model().nv, sim.model().nv);
    dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 1e-4);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 2e-3);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_ball_plane_gd)
{
    // This case tests gradient descent with a single sticking contact between a ball
    // and a plane. It minimizes the norm of contact forces wrt the applied torques.
    Model _model;
    GeometryModel _geom_model;
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const FrameIndex plane_frame = FrameIndex(0);
    const GeometryObject plane_geom = GeometryObject("plane", 0, plane_frame, SE3::Identity(), plane_ptr);
    GeomIndex plane_id = _geom_model.addGeometryObject(plane_geom);

    const std::string name = "ball_1";
    const std::string joint_name = "joint_1";
    const std::string frame_name = "frame_1";
    const double mass = 1.0;
    const double r = 1.0;
    const Inertia inertia = Inertia::FromSphere(mass, r);
    const JointModelFreeFlyer joint = JointModelFreeFlyer();
    const JointIndex parent = 0;
    const SE3 placement = SE3::Identity();
    JointIndex joint_id = _model.addJoint(parent, joint, placement, joint_name);
    _model.appendBodyToJoint(joint_id, inertia);
    coal::CollisionGeometryPtr_t sphere_ptr(new coal::Sphere(r));
    const GeometryObject sphere_geom = GeometryObject(name, joint_id, placement, sphere_ptr);
    GeomIndex box_id = _geom_model.addGeometryObject(sphere_geom);
    CollisionPair cp(plane_id, box_id);
    _geom_model.addCollisionPair(cp);

    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-15;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-15;

    // Testing sticking mode
    Eigen::VectorXd q = neutral(*model);
    q(2) = (r / 2.) * .8;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    const int max_gd_iters = 100;
    const double gd_step_size = 1e-1;

    std::vector<double> costs;
    costs.reserve(max_gd_iters + 1);

    // Run initial step to get starting contact forces
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd contact_forces = sim.workspace.constraint_problem().frictional_point_constraints_forces();
    costs.push_back(0.5 * contact_forces.squaredNorm());

    for (int iters = 0; iters < max_gd_iters; ++iters)
    {
        // Compute derivatives at current state
        SimulatorDerivatives dsim(sim);
        dsim.stepDerivatives(sim, q, v, tau, dt);
        // dlam/dtheta (columns correspond to generalized forces / tau)
        Eigen::MatrixXd dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);
        // gradient of 0.5 * ||lambda||^2 wrt tau is (dlambda/dtau)^T * lambda
        Eigen::VectorXd grad = dlam_dtau.transpose() * contact_forces;
        // gradient descent update on tau
        tau -= gd_step_size * grad;
        // step simulator with updated tau and read new contact forces
        step(sim, q, v, tau, dt);
        // sim.step<ADMM>(q, v, tau, dt);
        contact_forces = sim.workspace.constraint_problem().frictional_point_constraints_forces();
        const double cost = 0.5 * contact_forces.squaredNorm();
        costs.push_back(cost);
    }
    // Basic sanity: cost should have decreased (final < initial)
    BOOST_CHECK_LT(costs.back(), costs.front());
    // Check tau is opposite to gravity
    BOOST_CHECK_CLOSE(tau[2], 9.81 * mass, 1e-2);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_ball_plane_with_compliance)
{
    // This case tests derivatives with a single (both sliding and sticking) contact between a ball
    // and a plane.
    Model _model;
    GeometryModel _geom_model;
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const FrameIndex plane_frame = FrameIndex(0);
    GeometryObject plane_geom = GeometryObject("plane", 0, plane_frame, SE3::Identity(), plane_ptr);
    plane_geom.physicsMaterial.compliance = 0.1;
    GeomIndex plane_id = _geom_model.addGeometryObject(plane_geom);

    const std::string name = "ball_1";
    const std::string joint_name = "joint_1";
    const std::string frame_name = "frame_1";
    const double mass = 1.0;
    const double r = 1.0;
    const Inertia inertia = Inertia::FromSphere(mass, r);
    const JointModelFreeFlyer joint = JointModelFreeFlyer();
    const JointIndex parent = 0;
    const SE3 placement = SE3::Identity();
    JointIndex joint_id = _model.addJoint(parent, joint, placement, joint_name);
    _model.appendBodyToJoint(joint_id, inertia);
    coal::CollisionGeometryPtr_t sphere_ptr(new coal::Sphere(r));
    GeometryObject sphere_geom = GeometryObject(name, joint_id, placement, sphere_ptr);
    sphere_geom.physicsMaterial.compliance = 0.1;
    GeomIndex box_id = _geom_model.addGeometryObject(sphere_geom);
    CollisionPair cp(plane_id, box_id);
    _geom_model.addCollisionPair(cp);

    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    // Testing sticking mode
    Eigen::VectorXd q = neutral(*model);
    q(2) = (r / 2.) * .8;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd qnew = sim.state.qnew;
    Eigen::VectorXd vnew = sim.state.vnew;

    SimulatorDerivatives dsim(sim);
    dsim.stepDerivatives(sim, q, v, tau, dt);
    Eigen::MatrixXd dvnew_dq = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq = dsim.contact_solver_derivatives.dlam_dtheta().leftCols(sim.model().nv);
    Eigen::MatrixXd dlam_dv = dsim.contact_solver_derivatives.dlam_dtheta().middleCols(sim.model().nv, sim.model().nv);
    Eigen::MatrixXd dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);

    Eigen::MatrixXd dvnew_dq_fd = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv_fd = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau_fd = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq_fd = dlam_dq;
    Eigen::MatrixXd dlam_dv_fd = dlam_dv;
    Eigen::MatrixXd dlam_dtau_fd = dlam_dtau;

    SimulatorX sim_fd(model, geom_model);
    sim_fd.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim_fd.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim_fd.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    // std::cout << "q " << q << std::endl;
    // std::cout << "v " << v << std::endl;
    // std::cout << "qnew   " << qnew << std::endl;
    // std::cout << "vnew   " << vnew << std::endl;
    // std::cout << "contact forces" << sim.workspace.constraint_problem().frictional_point_constraints_forces();
    // for (int i = 0; i < sim.workspace.constraint_problem().getContactModes().size(); ++i)
    // {
    //   ConstraintProblem::ContactMode cat = sim.workspace.constraint_problem().getContactMode(i);
    //   switch (cat) {
    //   case ConstraintProblem::ContactMode::BREAKING:
    //       std::cout << "contact mode " << i << "  BREAKING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::SLIDING:
    //       std::cout << "contact mode " << i << "  SLIDING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::STICKING:
    //       std::cout << "contact mode " << i << "  STICKING" << std::endl;
    //   }
    // }
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 5e-3);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 1e-3);

    // testing sliding mode
    v(0) = 1;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    qnew = sim.state.qnew;
    vnew = sim.state.vnew;

    dsim.stepDerivatives(sim, q, v, tau, dt);
    dvnew_dq = dsim.dvnew_dq;
    dvnew_dv = dsim.dvnew_dv;
    dvnew_dtau = dsim.dvnew_dtau;
    dlam_dq = dsim.contact_solver_derivatives.dlam_dtheta().leftCols(sim.model().nv);
    dlam_dv = dsim.contact_solver_derivatives.dlam_dtheta().middleCols(sim.model().nv, sim.model().nv);
    dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 1e-4);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 2e-3);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_balls_plane)
{
    // This case tests derivatives with multiple balls on a plane. Balls are attached to a single
    // freeflyer joint (simulating a cube).
    Model _model;
    GeometryModel _geom_model;
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const FrameIndex plane_frame = FrameIndex(0);
    const GeometryObject plane_geom = GeometryObject("plane", 0, plane_frame, SE3::Identity(), plane_ptr);
    GeomIndex plane_id = _geom_model.addGeometryObject(plane_geom);

    const std::string name = "cube_";
    const std::string joint_name = "joint_";
    const std::string frame_name = "frame_";
    const double mass = 1.0;
    const double r = 1;
    std::vector<Eigen::Vector3d> balls_translation = {
        Eigen::Vector3d(r / 2, r / 2, -r / 2), Eigen::Vector3d(-r / 2, r / 2, -r / 2), Eigen::Vector3d(r / 2, -r / 2, -r / 2),
        Eigen::Vector3d(-r / 2, -r / 2, -r / 2)};
    const std::size_t N_balls = balls_translation.size();
    const Inertia inertia = Inertia::FromBox(mass, r, r, r);
    const JointModelFreeFlyer joint = JointModelFreeFlyer();
    const JointIndex parent = 0;
    const SE3 placement = SE3::Identity();
    JointIndex joint_id = _model.addJoint(parent, joint, placement, joint_name);
    _model.appendBodyToJoint(joint_id, inertia);
    for (std::size_t i = 0; i < N_balls; ++i)
    {
        const Eigen::Vector3d translation = balls_translation[static_cast<std::size_t>(i)];
        const std::string name_i = name + std::to_string(i);
        SE3 placement_i = SE3::Identity();
        placement_i.translation() = translation;
        coal::CollisionGeometryPtr_t sphere_ptr(new coal::Sphere(r / 10));
        const GeometryObject sphere_geom_i = GeometryObject(name_i, joint_id, placement_i, sphere_ptr);
        GeomIndex sphere_id_i = _geom_model.addGeometryObject(sphere_geom_i);
        CollisionPair cp(plane_id, sphere_id_i);
        _geom_model.addCollisionPair(cp);
    }

    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    // Testing sticking mode
    Eigen::VectorXd q = neutral(*model);
    q(2) = (r / 2.) * 0.8;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd qnew = sim.state.qnew;
    Eigen::VectorXd vnew = sim.state.vnew;

    SimulatorDerivatives dsim(sim);
    dsim.stepDerivatives(sim, q, v, tau, dt);
    Eigen::MatrixXd dvnew_dq = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq = dsim.contact_solver_derivatives.dlam_dtheta().leftCols(sim.model().nv);
    Eigen::MatrixXd dlam_dv = dsim.contact_solver_derivatives.dlam_dtheta().middleCols(sim.model().nv, sim.model().nv);
    Eigen::MatrixXd dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);

    Eigen::MatrixXd dvnew_dq_fd = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv_fd = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau_fd = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq_fd = dlam_dq;
    Eigen::MatrixXd dlam_dv_fd = dlam_dv;
    Eigen::MatrixXd dlam_dtau_fd = dlam_dtau;

    SimulatorX sim_fd(model, geom_model);
    sim_fd.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim_fd.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim_fd.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 1e-4);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 1e-4);

    // testing sliding mode
    v(0) = 1;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    qnew = sim.state.qnew;
    vnew = sim.state.vnew;

    dsim.stepDerivatives(sim, q, v, tau, dt);
    dvnew_dq = dsim.dvnew_dq;
    dvnew_dv = dsim.dvnew_dv;
    dvnew_dtau = dsim.dvnew_dtau;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    // std::cout << "q "  << q << std::endl;
    // std::cout << "v "  << v << std::endl;
    // std::cout << "qnew   " << qnew << std::endl;
    // std::cout << "vnew   " << vnew << std::endl;
    // std::cout << "contact forces" << sim.workspace.constraint_problem().frictional_point_constraints_forces();
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 1e-3);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    // EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 5e-3);
    // EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 5e-3);
    // EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 5e-3);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_colliding_balls)
{
    // This case tests derivatives with a single (sticking or sliding) contact between two balls.
    const int N_balls = 2;
    std::vector<double> radius(N_balls, .05);
    Model _model;
    GeometryModel _geom_model;

    const std::string name = "ball_1";
    const std::string joint_name = "joint_1";
    const std::string frame_name = "frame_1";
    const double mass = 1.0;
    const double r = radius[static_cast<std::size_t>(0)];
    const Inertia inertia = Inertia::FromSphere(mass, r);
    const JointModelFreeFlyer joint = JointModelFreeFlyer();
    const JointIndex parent = 0;
    const SE3 placement = SE3::Identity();
    JointIndex joint_id_1 = _model.addJoint(parent, joint, placement, joint_name);
    _model.appendBodyToJoint(joint_id_1, inertia);
    coal::CollisionGeometryPtr_t sphere_ptr(new coal::Sphere(r));
    const GeometryObject sphere_geom_1 = GeometryObject(name, joint_id_1, placement, sphere_ptr);
    GeomIndex sphere_id_1 = _geom_model.addGeometryObject(sphere_geom_1);
    JointIndex joint_id_2 = _model.addJoint(parent, joint, placement, joint_name);
    _model.appendBodyToJoint(joint_id_2, inertia);
    const GeometryObject sphere_geom_2 = GeometryObject(name, joint_id_2, placement, sphere_ptr);
    GeomIndex sphere_id_2 = _geom_model.addGeometryObject(sphere_geom_2);
    CollisionPair cp(sphere_id_1, sphere_id_2);
    _geom_model.addCollisionPair(cp);

    _model.gravity.setZero();
    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    // Testing sticking mode
    Eigen::VectorXd q = neutral(*model);
    q(7) = 0.06;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    v(0) = 1.0;
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    Eigen::VectorXd qnew = sim.state.qnew;
    Eigen::VectorXd vnew = sim.state.vnew;

    SimulatorDerivatives dsim(sim);
    dsim.stepDerivatives(sim, q, v, tau, dt);
    Eigen::MatrixXd dvnew_dq = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq = dsim.contact_solver_derivatives.dlam_dtheta().leftCols(sim.model().nv);
    Eigen::MatrixXd dlam_dv = dsim.contact_solver_derivatives.dlam_dtheta().middleCols(sim.model().nv, sim.model().nv);
    Eigen::MatrixXd dlam_dtau = dsim.contact_solver_derivatives.dlam_dtheta().rightCols(sim.model().nv);

    Eigen::MatrixXd dvnew_dq_fd = dsim.dvnew_dq;
    Eigen::MatrixXd dvnew_dv_fd = dsim.dvnew_dv;
    Eigen::MatrixXd dvnew_dtau_fd = dsim.dvnew_dtau;
    Eigen::MatrixXd dlam_dq_fd = dlam_dq;
    Eigen::MatrixXd dlam_dv_fd = dlam_dv;
    Eigen::MatrixXd dlam_dtau_fd = dlam_dtau;

    SimulatorX sim_fd(model, geom_model);
    sim_fd.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim_fd.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim_fd.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    // std::cout << "q "  << q << std::endl;
    // std::cout << "v "  << v << std::endl;
    // std::cout << "qnew   " << qnew << std::endl;
    // std::cout << "vnew   " << vnew << std::endl;
    // std::cout << "contact forces" << sim.workspace.constraint_problem().frictional_point_constraints_forces() << std::endl;
    // std::cout << "dvnew_dtau   " << dvnew_dtau << std::endl;
    // std::cout << "dvnew_dtau_fd   " << dvnew_dtau_fd << std::endl;
    // std::cout << "dvnew_dtau_fd - dvnew_dtau_  " << dvnew_dtau_fd - dvnew_dtau << std::endl;
    // std::cout << "dvnew_dv   " << dvnew_dv << std::endl;
    // std::cout << "dvnew_dv_fd   " << dvnew_dv_fd << std::endl;
    // std::cout << "dvnew_dq   " << dvnew_dq << std::endl;
    // std::cout << "dvnew_dq_fd   " << dvnew_dq_fd << std::endl;
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-3);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 1e-3);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 1e-4);
    // EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 5e-3);

    // testing sliding mode
    v(0) = 0.1;
    v(1) = 1;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    qnew = sim.state.qnew;
    vnew = sim.state.vnew;

    dsim.stepDerivatives(sim, q, v, tau, dt);
    dvnew_dq = dsim.dvnew_dq;
    dvnew_dv = dsim.dvnew_dv;
    dvnew_dtau = dsim.dvnew_dtau;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    // std::cout << "q "  << q << std::endl;
    // std::cout << "v "  << v << std::endl;
    // std::cout << "qnew   " << qnew << std::endl;
    // std::cout << "vnew   " << vnew << std::endl;
    // std::cout << "contact forces" << sim.workspace.constraint_problem().frictional_point_constraints_forces() << std::endl;
    // for (std::size_t i = 0; i < sim.workspace.constraint_problem().getContactModes().size(); ++i)
    // {
    //   ConstraintProblem::ContactMode cat = sim.workspace.constraint_problem().getContactMode(i);
    //   switch (cat) {
    //   case ConstraintProblem::ContactMode::BREAKING:
    //       std::cout << "contact mode " << i << "  BREAKING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::SLIDING:
    //       std::cout << "contact mode " << i << "  SLIDING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::STICKING:
    //       std::cout << "contact mode " << i << "  STICKING" << std::endl;
    //   }
    // }
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 5e-4);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    // EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 5e-3);
    // EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 5e-3);
    // EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 5e-3);

    // Testing another sliding mode
    // std::cout << "======================" << std::endl;
    // std::cout << "Testing another sliding mode" << std::endl;
    q(0) = 0.07;
    q(1) = 0.07;
    q(7) = 0.1;
    q(8) = 0.1;
    v(0) = 1;
    v(1) = 1;
    v(6) = 10;
    v(7) = -10;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    qnew = sim.state.qnew;
    vnew = sim.state.vnew;

    dsim.stepDerivatives(sim, q, v, tau, dt);
    dvnew_dq = dsim.dvnew_dq;
    dvnew_dv = dsim.dvnew_dv;
    dvnew_dtau = dsim.dvnew_dtau;

    computeStepDerivativesFD(sim_fd, q, v, tau, dt, dvnew_dq_fd, dvnew_dv_fd, dvnew_dtau_fd);
    // std::cout << "q "  << q << std::endl;
    // std::cout << "v "  << v << std::endl;
    // std::cout << "qnew   " << qnew << std::endl;
    // std::cout << "vnew   " << vnew << std::endl;
    // std::cout << "contact forces" << sim.workspace.constraint_problem().frictional_point_constraints_forces() << std::endl;
    // for (std::size_t i = 0; i < sim.workspace.constraint_problem().getContactModes().size(); ++i)
    // {
    //   ConstraintProblem::ContactMode cat = sim.workspace.constraint_problem().getContactMode(i);
    //   switch (cat) {
    //   case ConstraintProblem::ContactMode::BREAKING:
    //       std::cout << "contact mode " << i << "  BREAKING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::SLIDING:
    //       std::cout << "contact mode " << i << "  SLIDING" << std::endl;
    //       break;
    //   case ConstraintProblem::ContactMode::STICKING:
    //       std::cout << "contact mode " << i << "  STICKING" << std::endl;
    //   }
    // }
    EIGEN_VECTOR_IS_APPROX(dvnew_dtau, dvnew_dtau_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dv, dvnew_dv_fd, 1e-4);
    EIGEN_VECTOR_IS_APPROX(dvnew_dq, dvnew_dq_fd, 1e-2);
    computeLambdaDerivativesFD(sim_fd, q, v, tau, dt, dlam_dq_fd, dlam_dv_fd, dlam_dtau_fd);
    // EIGEN_VECTOR_IS_APPROX(dlam_dtau, dlam_dtau_fd, 5e-3);
    // EIGEN_VECTOR_IS_APPROX(dlam_dv, dlam_dv_fd, 5e-3);
    // EIGEN_VECTOR_IS_APPROX(dlam_dq, dlam_dq_fd, 5e-3);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_cube_plane)
{
    // This case tests derivatives with contact patches (both sticking and sliding) contacts
    // (cube/plane).
    Model _model;
    GeometryModel _geom_model;
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const FrameIndex plane_frame = FrameIndex(0);
    const GeometryObject plane_geom = GeometryObject("plane", 0, plane_frame, SE3::Identity(), plane_ptr);
    GeomIndex plane_id = _geom_model.addGeometryObject(plane_geom);

    const std::string name = "cube_";
    const std::string joint_name = "joint_";
    const std::string frame_name = "frame_";
    const double mass = 1.0;
    const double r = 1;
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

    std::shared_ptr<Model> model = std::make_shared<Model>(_model);
    std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    // Testing sticking mode
    Eigen::VectorXd q = neutral(*model);
    q(2) = r / 2.;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    SimulatorDerivatives dsim(sim);
    dsim.stepDerivatives(sim, q, v, tau, dt);

    // testing sliding mode
    q(2) = (r / 2.) * 0.8;
    v(0) = 1;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    dsim.stepDerivatives(sim, q, v, tau, dt);
}

BOOST_AUTO_TEST_CASE(simulator_derivatives_cubes_plane)
{
    // This case tests derivatives with contact patches and a mix of sliding and sticking contacts (2
    // cubes/plane).
    const int N_cubes = 2;
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
    SimulatorX sim(model, geom_model);
    sim.config.constraint_solvers_configs.admm_config.max_iter = 100;
    sim.config.constraint_solvers_configs.admm_config.absolute_precision = 1e-12;
    sim.config.constraint_solvers_configs.admm_config.relative_precision = 1e-12;

    Eigen::VectorXd q = neutral(*model);
    for (int i = 0; i < N_cubes; ++i)
    {
        q(i * 7) = static_cast<double>(i) * radius[static_cast<std::size_t>(i)] * 1.1;
        q(i * 7 + 2) = radius[static_cast<std::size_t>(i)] / 2.;
    }
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    v(1) = 3;
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    step(sim, q, v, tau, dt);
    // sim.step<ADMM>(q, v, tau, dt);
    SimulatorDerivatives dsim(sim);
    dsim.stepDerivatives(sim, q, v, tau, dt);
}

#endif // PINOCCHIO_WITH_HPP_FCL

BOOST_AUTO_TEST_SUITE_END()
