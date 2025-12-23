#include "simplex/core/simulator.hpp"
#include <pinocchio/algorithm/fwd.hpp>

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
#define ADMM ::pinocchio::ADMMContactSolverTpl
#define PGS ::pinocchio::PGSContactSolverTpl

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

void addFloorToGeomModel(GeometryModel &geom_model)
{
	using CollisionGeometryPtr = GeometryObject::CollisionGeometryPtr;

	CollisionGeometryPtr floor_collision_shape = CollisionGeometryPtr(new hpp::fcl::Halfspace(0.0, 0.0, 1.0, 0.0));

	const SE3 M = SE3::Identity();
	GeometryObject floor("floor", 0, 0, M, floor_collision_shape);
	geom_model.addGeometryObject(floor);
}

void addSystemCollisionPairs(const Model &model, GeometryModel &geom_model, const Eigen::VectorXd &qref)
{
	Data data(model);
	GeometryData geom_data(geom_model);
	// TI this function to gain compilation speed on this test
	::pinocchio::updateGeometryPlacements(model, data, geom_model, geom_data, qref);
	geom_model.removeAllCollisionPairs();
	for (std::size_t i = 0; i < geom_model.geometryObjects.size(); ++i)
	{
		for (std::size_t j = i; j < geom_model.geometryObjects.size(); ++j)
		{
			if (i == j)
			{
				continue; // don't add collision pair if same object
			}
			const GeometryObject &gobj_i = geom_model.geometryObjects[i];
			const GeometryObject &gobj_j = geom_model.geometryObjects[j];
			if (gobj_i.name == "floor" || gobj_j.name == "floor")
			{ // if floor, always add a collision pair
				geom_model.addCollisionPair(CollisionPair(i, j));
			}
			else
			{
				if (gobj_i.parentJoint == gobj_j.parentJoint)
				{ // don't add collision pair if same parent
					continue;
				}

				// run collision detection -- add collision pair if shapes are not colliding
				const SE3 M1 = geom_data.oMg[i];
				const SE3 M2 = geom_data.oMg[j];

				hpp::fcl::CollisionRequest colreq;
				colreq.security_margin = 1e-2; // 1cm of clearance
				hpp::fcl::CollisionResult colres;
				hpp::fcl::collide(
					gobj_i.geometry.get(), ::pinocchio::toFclTransform3f(M1), //
					gobj_j.geometry.get(), ::pinocchio::toFclTransform3f(M2), //
					colreq, colres);
				if (!colres.isCollision())
				{
					geom_model.addCollisionPair(CollisionPair(i, j));
				}
			}
		}
	}
}

BOOST_AUTO_TEST_CASE(mujoco_humanoid)
{
	ModelHandle model_handle(new Model());
	Model &model = ::pinocchio::helper::get_ref(model_handle);
	GeometryModelHandle geom_model_handle(new GeometryModel());
	GeometryModel &geom_model = ::pinocchio::helper::get_ref(geom_model_handle);

	const bool verbose = false;
	auto mjcf_path = findTestResource("SIMPLEX/tests/resources/mujoco_humanoid.xml");
	::pinocchio::mjcf::buildModel(mjcf_path, model, verbose);
	::pinocchio::mjcf::buildGeom(model, mjcf_path, pinocchio::COLLISION, geom_model);
	addFloorToGeomModel(geom_model);

	// sanity checks
	assert(model.nv == 27);
	assert(geom_model.geometryObjects.size() == 20);

	// initial state
	const Eigen::VectorXd q0 = model.referenceConfigurations["qpos0"];
	const Eigen::VectorXd v0 = Eigen::VectorXd::Zero(model.nv);
	const Eigen::VectorXd tau0 = Eigen::VectorXd::Zero(model.nv);

	// add collision pairs
	addSystemCollisionPairs(model, geom_model, q0);
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
	}
}

BOOST_AUTO_TEST_SUITE_END()
