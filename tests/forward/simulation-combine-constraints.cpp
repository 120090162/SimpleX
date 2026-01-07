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

/// \brief Creates a scene of 10 balls, all lying on a floor.
/// \param reduced_collision_pairs activate only the collision pair between balls and floor.
std::tuple<ModelHandle, GeometryModelHandle, Eigen::VectorXd>
createBallsScene(bool prismatic_joint, bool collision_plane, bool inter_colision)
{
    Model model;
    GeometryModel geom_model;

    // Add plane
    coal::CollisionGeometryPtr_t plane_ptr(new coal::Halfspace(0., 0., 1., 0.));
    const GeometryObject plane_geom = GeometryObject("plane", 0, SE3::Identity(), plane_ptr);
    GeomIndex plane_id = geom_model.addGeometryObject(plane_geom);

    // Add balls
    const std::size_t nballs = 10;
    const double radius = 1.0;
    const double mass = 1.0;
    for (std::size_t i = 0; i < nballs; ++i)
    {
        const std::string name = "ball_" + std::to_string(i);
        const std::string joint_name = "joint_" + std::to_string(i);
        const std::string frame_name = "frame_" + std::to_string(i);
        const Inertia inertia = Inertia::FromSphere(mass, radius);
        const JointIndex parent = 0;
        JointIndex joint_id;
        SE3 placement = SE3::Identity();
        ;
        if (prismatic_joint)
        {
            placement.translation() = Eigen::Vector3d(static_cast<double>(i) * 2 * radius * 1.1, 0, 0);
            const JointModelPZ joint = JointModelPZ();
            joint_id = model.addJoint(parent, joint, placement, joint_name);
        }
        else
        {
            const JointModelFreeFlyer joint = JointModelFreeFlyer();
            joint_id = model.addJoint(parent, joint, placement, joint_name);
        }
        model.appendBodyToJoint(joint_id, inertia);
        coal::CollisionGeometryPtr_t ball_ptr(new coal::Sphere(radius));
        const GeometryObject ball_geom = GeometryObject(name, joint_id, placement, ball_ptr);
        GeomIndex ball_id = geom_model.addGeometryObject(ball_geom);
        if (collision_plane)
        {
            CollisionPair col_pair(plane_id, ball_id);
            geom_model.addCollisionPair(col_pair);
        }
    }

    if (inter_colision)
    {
        for (size_t i = 0; i < geom_model.geometryObjects.size(); ++i)
        {
            for (size_t j = i + 1; j < geom_model.geometryObjects.size(); ++j)
            {
                CollisionPair col_pair(i, j);
                geom_model.addCollisionPair(col_pair);
            }
        }
    }

    Eigen::VectorXd q0 = neutral(model);
    for (int i = 0; i < (int)(nballs); ++i)
    {
        int nq = prismatic_joint ? 1 : 7;
        if (prismatic_joint)
        {
            q0(i * nq) = radius;
        }
        else
        {
            q0(i * nq) = static_cast<double>(i) * 2 * radius * 1.1;
            q0(i * nq + 2) = radius;
        }
    }

    return {
        std::make_shared<Model>(model),
        std::make_shared<GeometryModel>(geom_model),
        q0,
    };
}

void compareConstraintsProblems(const ConstraintsProblem & problem1, const ConstraintsProblem & problem2, const double tol)
{
    const std::size_t num_contacts = problem1.getNumberOfContacts();
    INDEX_EQUALITY_CHECK(num_contacts, problem2.getNumberOfContacts());
    if (num_contacts != problem2.getNumberOfContacts())
    {
        PINOCCHIO_THROW_PRETTY(std::runtime_error, "Cannot compare contact problems; They don't have the same amount of contacts.");
    }
    EIGEN_VECTOR_IS_APPROX(problem1.g(), problem2.g(), tol);
    // Need to compare the constraints cholesky up to `tol`
    // BOOST_CHECK(problem1.constraint_cholesky_decomposition == problem2.constraint_cholesky_decomposition);
    EIGEN_VECTOR_IS_APPROX(problem1.frictional_point_constraints_forces(), problem2.frictional_point_constraints_forces(), tol);
    for (std::size_t i = 0; i < num_contacts; ++i)
    {
        const Simulator::ConstraintModel & cmodel1 = problem1.constraint_models[i];
        const Simulator::ConstraintModel & cmodel2 = problem2.constraint_models[i];
        BOOST_CHECK(cmodel1 == cmodel2);
        BOOST_CHECK(problem1.previous_colliding_collision_pairs[i] == problem2.previous_colliding_collision_pairs[i]);
    }
}

double rand_interval(double rmin, double rmax)
{
    double val = rand() / (static_cast<double>(RAND_MAX) + 1);
    return (val * (rmax - rmin) + rmin);
}

// ---------------------------------------------------------------------------------------------------
// Actual tests
// ---------------------------------------------------------------------------------------------------
BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

// ---------------------------------------------------------------------------------------------------
// ----- Test stating scene
// Given an initial config that should be static test that the system is not moving.
void test_static_scene(bool prismatic_joint)
{
    // Get system with vertical prismatic, collision with floor and no inter collision
    std::tuple<ModelHandle, GeometryModelHandle, Eigen::VectorXd> system = createBallsScene(prismatic_joint, true, false);
    ModelHandle model = std::get<0>(system);
    GeometryModelHandle geom_model = std::get<1>(system);
    const Eigen::VectorXd & q0 = std::get<2>(system);

    // Initial state
    Eigen::VectorXd q = q0;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);

    // ... with this initial state, the balls should not move
    Simulator sim(model, geom_model);
    sim.admm_constraint_solver_settings.absolute_precision = 1e-12;
    sim.admm_constraint_solver_settings.relative_precision = 1e-12;
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);
    INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
    INDEX_EQUALITY_CHECK(sim.vnew.size(), (model->njoints - 1) * (prismatic_joint ? 1 : 6));
    INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
    const std::size_t num_contacts = sim.constraints_problem().getNumberOfContacts();
    INDEX_EQUALITY_CHECK(num_contacts, sim.geom_model().collisionPairs.size());
    BOOST_CHECK(sim.constraints_problem().check());
    BOOST_CHECK(sim.checkCollisionPairs());
    INDEX_EQUALITY_CHECK(sim.constraints_problem().g().size(), static_cast<Eigen::Index>(3 * num_contacts));
    EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-6);
    EIGEN_VECTOR_IS_APPROX(sim.vnew, v, 1e-6);
    EIGEN_VECTOR_IS_APPROX(sim.qnew, q, 1e-6);

    // Calling a second time step to test warmstart - contact solver should immediatly converge
    sim.step<ADMM>(q, v, tau, dt);
    // TODO: TEST - It gives 2 instead of 0 :( Warmstart must be solved
    INDEX_INEQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 2);

    // Running for 5 timesteps
    const std::size_t horizon = 5;
    q = q0;
    v.setZero();
    for (std::size_t i = 0; i < horizon; ++i)
    {
        sim.step<ADMM>(q, v, tau, dt);
        INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
        INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
        INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
        INDEX_EQUALITY_CHECK(sim.geom_model().collisionPairs.size(), sim.constraints_problem().constraint_models.size());
        const std::size_t num_contacts = sim.constraints_problem().getNumberOfContacts();
        INDEX_EQUALITY_CHECK(sim.geom_model().collisionPairs.size(), num_contacts);
        // BOOST_CHECK(sim.constraints_problem().check());
        INDEX_EQUALITY_CHECK(sim.constraints_problem().g().size(), static_cast<Eigen::Index>(3 * num_contacts));
        REAL_IS_APPROX(sim.vnew.norm(), 0.0, 1e-7);
        EIGEN_VECTOR_IS_APPROX(sim.vnew, v, 1e-8);
        EIGEN_VECTOR_IS_APPROX(sim.qnew, q, 1e-8);
        ConstraintsProblem problem1 = sim.constraints_problem();

        // Calling a second time step to test warmstart - contact solver should immediatly converge
        sim.step<ADMM>(q, v, tau, dt);
        // TODO: TEST - It gives 3 instead of 0 :( Warmstart must be solved
        INDEX_INEQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 3);
        INDEX_EQUALITY_CHECK(sim.constraints_problem().getNumberOfContacts(), num_contacts);
        ConstraintsProblem problem2 = sim.constraints_problem();
        compareConstraintsProblems(problem1, problem2, 1e-5);

        // Update state
        q = sim.qnew;
        v = sim.vnew;
    }
}
BOOST_AUTO_TEST_CASE(simulator_instance_static_balls_using_joint)
{
    test_static_scene(true);
}
BOOST_AUTO_TEST_CASE(simulator_instance_static_balls_using_freeflyer)
{
    test_static_scene(false);
}

// ---------------------------------------------------------------------------------------------------
// ----- Test moving scene
// Test if the simuation gives the same trajectory with two different simulator constructed differently
// and eventually with one using warmstart and other one do not
void test_moving_scene(bool prismatic_joint, bool compare_reset, bool inter_collision)
{
    // Get system
    std::tuple<ModelHandle, GeometryModelHandle, Eigen::VectorXd> system = createBallsScene(prismatic_joint, true, inter_collision);
    ModelHandle model = std::get<0>(system);
    GeometryModelHandle geom_model = std::get<1>(system);
    Eigen::VectorXd q0 = std::get<2>(system);
    Data _data(*model);
    std::shared_ptr<Data> data = std::make_shared<Data>(_data);
    GeometryData _geom_data(*geom_model);
    std::shared_ptr<GeometryData> geom_data = std::make_shared<GeometryData>(_geom_data);

    // Initial state
    const std::size_t nballs = geom_model->geometryObjects.size() - 1;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nballs); ++i)
    {
        Eigen::Index idx(prismatic_joint ? i : i * 7 + 2);
        q0(idx) += rand_interval(0.0, 1.0);
    }
    Eigen::VectorXd v0 = Eigen::VectorXd::Random(model->nv);

    std::array<Simulator, 2> sims{Simulator(model, geom_model), Simulator(model, data, geom_model, geom_data)};

    // Running for 5000 timesteps, twice
    const std::size_t horizon = 5;
    const double dt = 1e-3;
    std::array<std::vector<Eigen::VectorXd>, 2> qs;
    std::array<std::vector<Eigen::VectorXd>, 2> vs;
    std::array<std::vector<Eigen::VectorXd>, 2> vfrees;
    std::array<std::vector<ConstraintsProblem>, 2> constraints_problems;
    std::array<std::vector<int>, 2> numits;
    for (std::size_t i = 0; i < qs.size(); ++i)
    {
        Eigen::VectorXd q = q0;
        Eigen::VectorXd v = v0;
        Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
        qs[i].reserve(horizon);
        vs[i].reserve(horizon);
        vfrees[i].reserve(horizon);
        constraints_problems[i].reserve(horizon);
        numits[i].reserve(horizon);

        Simulator & sim = sims[i];
        if (compare_reset && i == 1)
        {
            sim.reset();
        }

        for (std::size_t t = 0; t < horizon; ++t)
        {
            sim.step<ADMM>(q, v, tau, dt);
            INDEX_EQUALITY_CHECK(sim.vfree.size(), model->nv);
            INDEX_EQUALITY_CHECK(sim.vnew.size(), model->nv);
            INDEX_EQUALITY_CHECK(sim.ftotal.size(), static_cast<std::size_t>(model->njoints));
            // BOOST_CHECK(sim.constraints_problem().check());

            // Update state
            q = sim.qnew;
            v = sim.vnew;

            qs[i].emplace_back(q);
            vs[i].emplace_back(v);
            vfrees[i].emplace_back(sim.vfree);
            constraints_problems[i].emplace_back(sim.constraints_problem());
            numits[i].emplace_back(sim.admm_constraint_solver.getIterationCount());
        }
    }

    // Check that trajectories are identical
    for (std::size_t i = 0; i < horizon; ++i)
    {
        const double tol = 1e-10;
        EIGEN_VECTOR_IS_APPROX(qs[0][i], qs[1][i], tol);
        EIGEN_VECTOR_IS_APPROX(vs[0][i], vs[1][i], tol);
        EIGEN_VECTOR_IS_APPROX(vfrees[0][i], vfrees[1][i], tol);
        compareConstraintsProblems(constraints_problems[0][i], constraints_problems[1][i], tol);
        INDEX_EQUALITY_CHECK(numits[0][i], numits[1][i]);
    }
}
BOOST_AUTO_TEST_CASE(simulator_instance_moving_balls_using_joint)
{
    test_moving_scene(true, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_moving_balls_using_joint_compare_warmstart)
{
    test_moving_scene(true, true, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_moving_balls_using_freeflyer)
{
    test_moving_scene(false, false, true);
}
BOOST_AUTO_TEST_CASE(simulator_instance_moving_balls_using_freeflyer_compare_warmstart)
{
    test_moving_scene(false, true, true);
}
BOOST_AUTO_TEST_CASE(simulator_instance_moving_balls_using_freeflyer_nointercollision)
{
    test_moving_scene(false, false, false);
}

// ---------------------------------------------------------------------------------------------------
// ----- Test one step with different combination of constraints
// Test if one step give the expected result and the right constraint problem size
void test_constraint_scene(bool prismatic_joint, bool floor_contact, bool bilat, bool weld, bool friction, bool limit, bool rand)
{
    // Get system
    std::tuple<ModelHandle, GeometryModelHandle, Eigen::VectorXd> system = createBallsScene(prismatic_joint, floor_contact, false);
    ModelHandle model = std::get<0>(system);
    GeometryModelHandle geom_model = std::get<1>(system);
    Eigen::VectorXd & q0 = std::get<2>(system);
    const std::size_t nballs = (std::size_t)(model->njoints - 1);
    DataHandle data = std::make_shared<Data>(*model);

    // Add joint constraints
    // Quantities for joint constraints
    // TODO: TEST - Update when nq>1 is available
    bool really_friction = friction && prismatic_joint;
    bool really_limit = limit && prismatic_joint;
    std::size_t n_active_limits = 0;
    const std::size_t n_constrained_v = nballs;
    if (really_friction)
    {
        model->lowerDryFrictionLimit = Eigen::VectorXd::Ones(model->nv) * -1.0;
        model->upperDryFrictionLimit = Eigen::VectorXd::Ones(model->nv) * 1.0;
    }
    if (really_limit)
    {
        model->lowerPositionLimit = Eigen::VectorXd::Ones(model->nv) * -100000.0;
        model->upperPositionLimit = Eigen::VectorXd::Ones(model->nv) * 100000.0;
    }

    // Initial state
    Eigen::VectorXd q = q0;
    Eigen::VectorXd v = Eigen::VectorXd::Zero(model->nv);
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(model->nv);
    if (rand)
    {
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(nballs); ++i)
        {
            Eigen::Index idx(prismatic_joint ? i : i * 7 + 2);
            q[idx] += rand_interval(0.0, 1.0);
            if (q[idx] - model->positionLimitMargin[idx] <= model->lowerPositionLimit(idx))
            {
                n_active_limits++;
            }
            if (q[idx] + model->positionLimitMargin[idx] >= model->upperPositionLimit(idx))
            {
                n_active_limits++;
            }
        }
        v = Eigen::VectorXd::Random(model->nv);
    }

    // Adding one attach constraint for each ball if required
    pinocchio::forwardKinematics(*model, *data, q);
    WeldConstraintModelVector weld_constraint_models;
    BilateralPointConstraintModelVector bilateral_constraint_models;
    std::size_t end_bilat = 0;
    std::size_t start_weld = nballs;
    std::size_t attach_constraint_size = 0;
    if (bilat && weld)
    {
        // Half of balls are welded, other are point attached
        end_bilat = nballs / 2;
        start_weld = nballs / 2;
        attach_constraint_size = (nballs / 2) * 3 + (nballs - nballs / 2) * 6;
    }
    else if (bilat && !weld)
    {
        attach_constraint_size = nballs * 3;
        end_bilat = nballs;
    }
    else if (!bilat && weld)
    {
        attach_constraint_size = nballs * 6;
        start_weld = 0;
    }
    else if (!bilat && !weld)
    {
        attach_constraint_size = 0;
    }
    for (std::size_t i = 0; i < end_bilat; ++i)
    {
        const std::string name = "joint_" + std::to_string(i);
        const JointIndex joint1_id = model->getJointId(name);
        const GeomIndex joint2_id = 0;
        const SE3 joint1_placement = SE3::Identity();
        const SE3 joint2_placement = data->oMi[joint1_id];
        bilateral_constraint_models.push_back(
            BilateralPointConstraintModel(*model, joint1_id, joint1_placement, joint2_id, joint2_placement));
    }
    for (std::size_t i = start_weld; i < nballs; ++i)
    {
        const std::string name = "joint_" + std::to_string(i);
        const JointIndex joint1_id = model->getJointId(name);
        const GeomIndex joint2_id = 0;
        const SE3 joint1_placement = SE3::Identity();
        const SE3 joint2_placement = data->oMi[joint1_id];
        weld_constraint_models.push_back(WeldConstraintModel(*model, joint1_id, joint1_placement, joint2_id, joint2_placement));
    }

    // Creating the simulator and make a step
    Simulator sim(model, geom_model, bilateral_constraint_models, weld_constraint_models);
    const double dt = 1e-3;
    sim.step<ADMM>(q, v, tau, dt);

    // Test that if balls are on the floor there is 1 contact by ball
    bool is_laying = !rand && floor_contact;
    std::size_t ncontacts = 0;
    if (is_laying)
    {
        ncontacts = nballs;
        INDEX_EQUALITY_CHECK(ncontacts, sim.constraints_problem().getNumberOfContacts());
    }

    // Test if that ball should not move the new value are 0
    bool must_not_move = is_laying || (!rand && bilat) || (weld && !bilat);
    if (must_not_move)
    {
        // TODO: TEST - Here precision of 1e-5 is required.. Maybe lack of precision ?
        EIGEN_VECTOR_IS_APPROX(sim.vnew, Eigen::VectorXd::Zero(model->nv), 1e-5);
        EIGEN_VECTOR_IS_APPROX(sim.qnew, q, 1e-5);
        if (!rand)
        {
            EIGEN_VECTOR_IS_APPROX(sim.vnew, v, 1e-5);
        }
    }

    // Test the constraint problem size
    INDEX_EQUALITY_CHECK(
        sim.constraints_problem().constraints_problem_size(),
        (int)(3 * ncontacts + attach_constraint_size + really_friction * n_constrained_v + really_limit * n_active_limits));

    // calling the simulator again should not change the state
    // and the solver should converge in one iteration
    if (must_not_move && !rand)
    {
        sim.step<ADMM>(q, v, tau, dt);
        // TODO: TEST - It gives many different results instead of 0 :( Warmstart must be solved
        // Uperbound bound 20 is required
        INDEX_INEQUALITY_CHECK(sim.admm_constraint_solver.getIterationCount(), 20);
    }
}

// test_constraint_scene(
//   prismatic_joint,
//   floor_contact,
//   bilat,
//   weld,
//   friction,
//   limit,
//   rand
// )
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_no_constraint)
{
    test_constraint_scene(false, true, false, false, false, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_bilateral_constraint)
{
    test_constraint_scene(false, false, true, false, false, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_weld_constraint)
{
    test_constraint_scene(false, false, false, true, false, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_weld_constraint_ill_defined)
{
    test_constraint_scene(false, false, false, true, false, false, true);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_weld_and_bilat_constraint)
{
    test_constraint_scene(false, false, true, true, false, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_limit)
{
    test_constraint_scene(true, false, false, false, false, true, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_friction)
{
    test_constraint_scene(true, false, false, false, true, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_limit_static)
{
    test_constraint_scene(true, true, false, false, false, true, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_friction_static)
{
    test_constraint_scene(true, true, false, false, true, false, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_all_joints_constraint_static)
{
    test_constraint_scene(true, true, false, false, true, true, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_all_constraints)
{
    test_constraint_scene(true, false, true, true, true, true, true);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_all_constraints_static)
{
    test_constraint_scene(true, false, true, true, true, true, false);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_all_constraints_and_contact)
{
    test_constraint_scene(true, true, true, true, true, true, true);
}
BOOST_AUTO_TEST_CASE(simulator_instance_step_with_all_constraints_and_contact_static)
{
    test_constraint_scene(true, true, true, true, true, true, false);
}
// Loop on all combination
BOOST_AUTO_TEST_CASE(simulator_instance_step_test_all_combination)
{
    for (std::size_t i = 0; i < 128; ++i)
    {
        bool prismatic_joint = i & 0x01;
        bool floor_contact = (i >> 1) & 0x01;
        bool bilat = (i >> 2) & 0x01;
        bool weld = (i >> 3) & 0x01;
        bool friction = (i >> 4) & 0x01;
        bool limit = (i >> 5) & 0x01;
        bool rand = (i >> 6) & 0x01;
        test_constraint_scene(prismatic_joint, floor_contact, bilat, weld, friction, limit, rand);
    }
}

BOOST_AUTO_TEST_SUITE_END()
