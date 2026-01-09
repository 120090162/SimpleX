#include "simplex/core/simulator-x.hpp"
#include "simplex/core/ncp-derivatives.hpp"

#include <pinocchio/algorithm/fwd.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/sample-models.hpp>

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

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(contact_solver_derivatives_constructor)
{
    ModelHandle model(new Model());
    DataHandle data(new Data(*model));
    GeometryModelHandle geom_model(new GeometryModel());
    GeometryDataHandle geom_data(new GeometryData(*geom_model));
    SimulatorX sim(model, data, geom_model, geom_data);
    BOOST_CHECK_NO_THROW(ContactSolverDerivatives dcontact(sim.workspace.getConstraintProblemHandle()));
}

#ifdef PINOCCHIO_WITH_HPP_FCL
BOOST_AUTO_TEST_CASE(contact_solver_derivatives_sticking_cube)
{
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
    Data _data(_model);
    std::shared_ptr<Data> data = std::make_shared<Data>(_data);
    GeometryData _geom_data(_geom_model);
    std::shared_ptr<GeometryData> geom_data = std::make_shared<GeometryData>(_geom_data);
    SimulatorX sim(model, data, geom_model, geom_data);

    Eigen::VectorXd q = neutral(*model);

    q(0) = r * 1.1;
    q(2) = r / 2.;

    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    const double dt = 1e-3;
    sim.step(q, v, tau, dt);
    //  TODO: Add tests on sticking and sliding contacts
    ContactSolverDerivatives dcontact(sim.workspace.getConstraintProblemHandle());
    int ndtheta = model->nv;
    dcontact.compute();
    Eigen::MatrixXd dGlamgdtheta =
        Eigen::MatrixXd::Zero((Eigen::Index)(3 * sim.workspace.constraint_problem().getNumberOfContacts()), ndtheta);
    dcontact.jvp(::pinocchio::make_const_ref(dGlamgdtheta));
}

BOOST_AUTO_TEST_CASE(contact_solver_derivatives_sliding_and_sticking_cubes)
{
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
    Data _data(_model);
    std::shared_ptr<Data> data = std::make_shared<Data>(_data);
    GeometryData _geom_data(_geom_model);
    std::shared_ptr<GeometryData> geom_data = std::make_shared<GeometryData>(_geom_data);
    SimulatorX sim(model, data, geom_model, geom_data);

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
    sim.step(q, v, tau, dt);
    ContactSolverDerivatives dcontact(sim.workspace.getConstraintProblemHandle());
    int ndtheta = model->nv;
    dcontact.compute();
    Eigen::MatrixXd dGlamgdtheta =
        Eigen::MatrixXd::Zero((Eigen::Index)(3 * sim.workspace.constraint_problem().getNumberOfContacts()), ndtheta);
    dcontact.jvp(::pinocchio::make_const_ref(dGlamgdtheta));
}
#endif // PINOCCHIO_WITH_HPP_FCL

BOOST_AUTO_TEST_SUITE_END()
