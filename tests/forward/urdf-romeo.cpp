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

#ifndef PINOCCHIO_MODEL_DIR
#define PINOCCHIO_MODEL_DIR std::string(SIMPLEX_TEST_DIR) + "/third_party/pinocchio/models"
#endif

using ModelHandle = Simulator::ModelHandle;
using GeometryModelHandle = Simulator::GeometryModelHandle;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(simulator_instance_romeo_step)
{
	const std::string urdf_filename = PINOCCHIO_MODEL_DIR + std::string("/example-robot-data/robots/romeo_description/urdf/romeo_small.urdf");
	const std::string srdf_filename = PINOCCHIO_MODEL_DIR + std::string("/example-robot-data/robots/romeo_description/srdf/romeo.srdf");
	std::vector<std::string> package_dirs;
	const std::string mesh_dir = PINOCCHIO_MODEL_DIR;
	package_dirs.push_back(mesh_dir);

	Model _model;
	pinocchio::urdf::buildModel(urdf_filename, pinocchio::JointModelFreeFlyer(), _model);
	GeometryModel _geom_model;
	pinocchio::urdf::buildGeom(_model, urdf_filename, pinocchio::COLLISION, _geom_model, package_dirs);
	_geom_model.addAllCollisionPairs();
	pinocchio::srdf::removeCollisionPairs(_model, _geom_model, srdf_filename, false);

	Data _data(_model);
	GeometryData _geom_data(_geom_model);

	pinocchio::srdf::loadReferenceConfigurations(_model, srdf_filename, false);
	Eigen::VectorXd q = _model.referenceConfigurations["half_sitting"];
	Eigen::VectorXd v = Eigen::VectorXd::Zero(_model.nv);
	Eigen::VectorXd tau = Eigen::VectorXd::Zero(_model.nv);
	const double dt = 1e-3;

	std::shared_ptr<Model> model = std::make_shared<Model>(_model);
	std::shared_ptr<Data> data = std::make_shared<Data>(_data);
	std::shared_ptr<GeometryModel> geom_model = std::make_shared<GeometryModel>(_geom_model);
	std::shared_ptr<GeometryData> geom_data = std::make_shared<GeometryData>(_geom_data);
	Simulator sim(model, data, geom_model, geom_data);
	BOOST_CHECK_NO_THROW(sim.step<ADMM>(q, v, tau, dt));
	BOOST_CHECK(sim.vfree.size() == model->nv);
	BOOST_CHECK(sim.vnew.size() == model->nv);
	BOOST_CHECK(sim.ftotal.size() == static_cast<std::size_t>(model->njoints));
}

BOOST_AUTO_TEST_SUITE_END()
