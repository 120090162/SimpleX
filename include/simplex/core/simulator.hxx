//
// Copyright (c) 2024 INRIA
//

#ifndef __simple_core_simulator_hxx__
#define __simple_core_simulator_hxx__

#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/contact-jacobian.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/contact-inverse-dynamics.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <pinocchio/collision/collision.hpp>
#include <pinocchio/collision/broadphase.hpp>

#include "simple/tracy.hpp"

// #include <pinocchio/algorithm/contact-jacobian.hpp>

#include <pinocchio/multibody/fwd.hpp>

#include <hpp/fcl/collision_data.h>

#include <boost/variant.hpp>

namespace simple
{
  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle,
    DataHandle data_handle,
    GeometryModelHandle geom_model_handle,
    GeometryDataHandle geom_data_handle,
    const BilateralPointConstraintModelVector & bilateral_constraint_models,
    const WeldConstraintModelVector & weld_constraint_models)
  : m_model(model_handle)
  , m_data(data_handle)
  , m_geom_model(geom_model_handle)
  , m_geom_data(geom_data_handle)
  , q(this->model().nq)
  , v(this->model().nv)
  , tau(this->model().nv)
  , fext(static_cast<std::size_t>(this->model().njoints), Force::Zero())
  , dt(-1)
  , qnew(this->model().nq)
  , vfree(this->model().nv)
  , vnew(this->model().nv)
  , anew(this->model().nv)
  , ftotal(static_cast<std::size_t>(this->model().njoints), Force::Zero())
  , tau_total(this->model().nv)
  , tau_constraints(this->model().nv)
  , constraint_solver_type(ConstraintSolverType::NONE)
  , admm_constraint_solver(0) // only create it when needed
  , pgs_constraint_solver(0)  // only create it when needed
  , m_broad_phase_manager(
      new BroadPhaseManager(&helper::get_ref(model_handle), &helper::get_ref(geom_model_handle), &helper::get_ref(geom_data_handle)))
  , m_collision_callback(this->geom_model(), this->geom_data())
  , m_constraints_problem(new ConstraintsProblem(
      model_handle, data_handle, geom_model_handle, geom_data_handle, bilateral_constraint_models, weld_constraint_models))
  , m_vnew_integration_tmp(this->model().nv)
  , m_is_reset(true)
  , m_step_timer(false)
  , m_internal_timer(false)
  {
    this->initializeGeometryData();
    this->allocate();
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle,
    DataHandle data_handle,
    GeometryModelHandle geom_model_handle,
    GeometryDataHandle geom_data_handle,
    const BilateralPointConstraintModelVector & bilateral_constraint_models)
  : SimulatorTpl(model_handle, data_handle, geom_model_handle, geom_data_handle, bilateral_constraint_models, WeldConstraintModelVector())
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle,
    DataHandle data_handle,
    GeometryModelHandle geom_model_handle,
    GeometryDataHandle geom_data_handle,
    const WeldConstraintModelVector & weld_constraint_models)
  : SimulatorTpl(
      model_handle, data_handle, geom_model_handle, geom_data_handle, BilateralPointConstraintModelVector(), weld_constraint_models)
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle, DataHandle data_handle, GeometryModelHandle geom_model_handle, GeometryDataHandle geom_data_handle)
  : SimulatorTpl(
      model_handle, data_handle, geom_model_handle, geom_data_handle, BilateralPointConstraintModelVector(), WeldConstraintModelVector())
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle,
    GeometryModelHandle geom_model_handle,
    const BilateralPointConstraintModelVector & bilateral_constraint_models,
    const WeldConstraintModelVector & weld_constraint_models)
  : SimulatorTpl(
      model_handle,
      std::make_shared<::pinocchio::Data>(*model_handle),
      geom_model_handle,
      std::make_shared<::pinocchio::GeometryData>(*geom_model_handle),
      bilateral_constraint_models,
      weld_constraint_models)
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle,
    GeometryModelHandle geom_model_handle,
    const BilateralPointConstraintModelVector & bilateral_constraint_models)
  : SimulatorTpl(model_handle, geom_model_handle, bilateral_constraint_models, WeldConstraintModelVector())
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(
    ModelHandle model_handle, GeometryModelHandle geom_model_handle, const WeldConstraintModelVector & weld_constraint_models)
  : SimulatorTpl(model_handle, geom_model_handle, BilateralPointConstraintModelVector(), weld_constraint_models)
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  SimulatorTpl<S, O, JointCollectionTpl>::SimulatorTpl(ModelHandle model_handle, GeometryModelHandle geom_model_handle)
  : SimulatorTpl(model_handle, geom_model_handle, BilateralPointConstraintModelVector(), WeldConstraintModelVector())
  {
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void SimulatorTpl<S, O, JointCollectionTpl>::allocate()
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::allocate");

    this->constraints_problem().allocate();
    //
    this->q.resize(this->model().nq);
    this->v.resize(this->model().nv);
    this->tau.resize(this->model().nv);
    this->fext.resize(static_cast<std::size_t>(model().njoints), Force::Zero());
    //
    this->qnew.resize(this->model().nq);
    this->vfree.resize(this->model().nv);
    this->vnew.resize(this->model().nv);
    this->anew.resize(this->model().nv);
    this->anew.setZero();
    this->ftotal.resize(static_cast<std::size_t>(model().njoints), Force::Zero());
    this->tau_total.resize(this->model().nv);
    this->tau_total.setZero();
    this->tau_constraints.resize(this->model().nv);
    this->tau_constraints.setZero();
    this->m_vnew_integration_tmp.resize(this->model().nv);
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void SimulatorTpl<S, O, JointCollectionTpl>::initializeGeometryData()
  {
    // Broad phase setup
    for (::pinocchio::GeometryObject & geom : this->geom_model().geometryObjects)
    {
      geom.geometry->computeLocalAABB();
    }

    // Narrow phase setup
    for (hpp::fcl::CollisionRequest & request : this->geom_data().collisionRequests)
    {
      request.enable_contact = true;
    }
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void SimulatorTpl<S, O, JointCollectionTpl>::reset()
  {
    // Reset constraint problem
    this->constraints_problem().clear();
    this->constraints_problem().constraints_forces().setZero();
    this->constraints_problem().previous_constraints_forces().setZero();

    // Reset constraint solver
    this->constraint_solver_type = ConstraintSolverType::NONE;

    // Reset timings
    this->m_step_timings.clear();
    this->m_constraint_solver_timings.clear();

    this->m_is_reset = true;
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  template<template<typename> class ConstraintSolver, typename ConfigVectorType, typename VelocityVectorType, typename TorqueVectorType>
  void SimulatorTpl<S, O, JointCollectionTpl>::step(
    const Eigen::MatrixBase<ConfigVectorType> & q,
    const Eigen::MatrixBase<VelocityVectorType> & v,
    const Eigen::MatrixBase<TorqueVectorType> & tau,
    Scalar dt)
  {
    this->fext.assign((std::size_t)(model().njoints), Force::Zero());
    this->step<ConstraintSolver>(q, v, tau, this->fext, dt);
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  template<
    template<typename>
    class ConstraintSolver,
    typename ConfigVectorType,
    typename VelocityVectorType,
    typename TorqueVectorType,
    typename ForceDerived>
  void SimulatorTpl<S, O, JointCollectionTpl>::step(
    const Eigen::MatrixBase<ConfigVectorType> & q_,
    const Eigen::MatrixBase<VelocityVectorType> & v_,
    const Eigen::MatrixBase<TorqueVectorType> & tau_,
    const ::pinocchio::container::aligned_vector<ForceDerived> & fext_,
    const Scalar dt_)
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::step");
    // clang-format off
    if (this->measure_timings) { this->m_step_timer.start(); };
    // clang-format on

    // TODO(louis): should we use check arguments or assert/throw?
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      this->vfree.size(), this->model().nv,
      "The sizes of the free velocity of the simulator and the input velocity do not match. "
      "You problably changed your model, data, geom_model or geom_data and forgot to call "
      "allocate().");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(q_.size(), this->model().nq, "The joint configuration vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(v_.size(), this->model().nv, "The joint velocity vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(tau_.size(), this->model().nv, "The joint torque vector is not of right size");
    PINOCCHIO_CHECK_ARGUMENT_SIZE(
      fext_.size(), static_cast<std::size_t>(this->model().njoints), "The external forces vector is not of right size");
    PINOCCHIO_CHECK_INPUT_ARGUMENT(dt_ >= 0, "dt is not >= 0");
    assert(this->check() && "The simulator is not properly instanciated.");

    // Record state of simulator
    this->q = q_;
    this->v = v_;
    this->tau = tau_;
    this->fext = fext_;
    this->dt = dt_;

    // Set up data for downstream algorithms
    this->data().q_in = q;
    this->data().v_in = v;
    this->data().tau_in = tau;

    // Compute the mass matrix of the system - used by the delassus operator
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::step - compute crba");
      ::pinocchio::crba(this->model(), this->data(), q, pinocchio::Convention::WORLD);
    }

    // Compute free acceleration of the system
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::step - compute vfree (first aba)");
      this->ftotal = fext;
      this->tau_total = tau;
      // clang-format off
      this->vfree = v + dt * ::pinocchio::aba(this->model(), this->data(), q, v, this->tau_total, this->ftotal, pinocchio::Convention::WORLD);
      // clang-format on
    }

    // Collision detection
    this->detectCollisions();

    // Update constraint problem with result of collision detection
    const bool compute_constraints_forces_warm_start = this->warm_start_constraints_forces && (!this->isReset());
    this->constraints_problem().update(compute_constraints_forces_warm_start);

    this->tau_constraints.setZero();
    if (this->constraints_problem().constraints_problem_size() > 0)
    {
      // Constraint resolution - compute the total joint forces (if there are any collisions) i.e. external forces + constraint forces
      this->resolveConstraints<ConstraintSolver>(v, dt);

      {
        SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::step - compute vnew (second aba)");
        this->anew = ::pinocchio::aba(this->model(), this->data(), q, v, this->tau_total, this->ftotal, pinocchio::Convention::WORLD);
        this->vnew = v + dt * this->anew;
      }
    }
    else
    {
      this->anew = (this->vfree - v) / dt; // TODO(quentin) do it differrently
      this->vnew = this->vfree;

      // Reset constraint solver timings
      this->m_constraint_solver_timings.clear();
    }

    this->m_vnew_integration_tmp = this->vnew * dt;
    ::pinocchio::integrate(this->model(), q, this->m_vnew_integration_tmp, this->qnew);
    this->m_is_reset = false;

    // clang-format off
    if (this->measure_timings) { this->m_step_timer.stop(); };
    this->m_step_timings = this->m_step_timer.elapsed();
    // clang-format on
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void SimulatorTpl<S, O, JointCollectionTpl>::detectCollisions()
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::detectCollisions");

    // clang-format off
    if (this->measure_timings) { this->m_internal_timer.start(); }
    // clang-format on

    // Compute oMg for each geometry
    ::pinocchio::updateGeometryPlacements(this->model(), this->data(), this->geom_model(), this->geom_data());

    // Reset collision results - super important! Otherwise constraint from previous time step may be
    // detected between non-colliding geometries.
    for (hpp::fcl::CollisionResult & col_res : this->geom_data().collisionResults)
    {
      col_res.clear();
    }

    // Run broad + narrow phase collision detection
    const bool recompute_local_aabb = false; // already computed in `initializeGeometryData`
    this->m_broad_phase_manager->update(recompute_local_aabb);
    assert(this->m_broad_phase_manager->check() && "The broad phase manager is not aligned with the geometry model.");

    // --> Broad Phase
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::detectCollisions - Broad phase");
      ::pinocchio::computeCollisions(*(this->m_broad_phase_manager), &this->m_collision_callback);
    }

    // --> Narrow Phase
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::detectCollisions - Narrow phase");
      for (std::size_t i = 0; i < this->m_collision_callback.pair_indexes.size(); ++i)
      {
        const std::size_t pair_index = this->m_collision_callback.pair_indexes[i];
        const ::pinocchio::CollisionPair & cp = this->geom_model().collisionPairs[pair_index];
        const ::pinocchio::GeometryObject & obj1 = this->geom_model().geometryObjects[cp.first];
        const ::pinocchio::GeometryObject & obj2 = this->geom_model().geometryObjects[cp.second];

        // middle phase collision detection
        coal::CollisionRequest collision_request(this->geom_data().collisionRequests[pair_index]);
        bool obb_overlap = false;
        if (
          obj1.geometry->getNodeType() == coal::GEOM_PLANE || obj1.geometry->getNodeType() == coal::GEOM_HALFSPACE
          || obj2.geometry->getNodeType() == coal::GEOM_PLANE || obj2.geometry->getNodeType() == coal::GEOM_HALFSPACE)
        {
          obb_overlap = true;
        }
        else
        {
          coal::Transform3s oM1(toFclTransform3f(this->geom_data().oMg[cp.first]));
          coal::Transform3s oM2(toFclTransform3f(this->geom_data().oMg[cp.second]));
          const Scalar security_margin = collision_request.security_margin;
          //
          const coal::AABB aabb1 = obj1.geometry->aabb_local.expand(security_margin * 0.5);
          coal::OBB obb1;
          coal::convertBV(aabb1, oM1, obb1);
          //
          const coal::AABB aabb2 = obj1.geometry->aabb_local.expand(security_margin * 0.5);
          coal::OBB obb2;
          coal::convertBV(aabb2, oM2, obb2);
          obb_overlap = obb1.overlap(obb2);
        }
        try
        {
          if (obb_overlap)
          {
            pinocchio::computeCollision(this->geom_model(), this->geom_data(), pair_index, collision_request);
            ::pinocchio::computeContactPatch(this->geom_model(), this->geom_data(), pair_index);
          }
        }
        catch (std::logic_error & e)
        {
          PINOCCHIO_THROW_PRETTY(
            std::logic_error, "Geometries with index go1: " << cp.first << " or go2: " << cp.second
                                                            << " have produced an internal error within Coal.\n what:\n"
                                                            << e.what());
        }
      }
    }

    // clang-format off
    if (this->measure_timings) { this->m_internal_timer.start(); }
    this->m_collision_detection_timings = this->m_internal_timer.elapsed();
    // clang-format on
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  template<template<typename> class ConstraintSolver, typename VelocityVectorType>
  void SimulatorTpl<S, O, JointCollectionTpl>::resolveConstraints(const Eigen::MatrixBase<VelocityVectorType> & v, const Scalar dt)
  {
    SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::resolveConstraints");

    assert(
      !this->constraints_problem().constraint_models.empty() && "Resolve collisions should not be called if there are no constraints.");

    // Build the constraint quantities for the constraint solver
    // TODO(quentin): warm-start constraint forces via constraint inverse dynamics
    this->constraints_problem().build(this->vfree, v, v + dt * this->anew, dt);
    this->preambleResolveConstraints(dt);

    // Call the constraint solver
    // clang-format off
    if (this->measure_timings) { this->m_internal_timer.start(); };
    details::SimulatorConstraintSolverTpl<ConstraintSolver, S, O, JointCollectionTpl>::run(*this, dt);
    if (this->measure_timings) { this->m_internal_timer.stop(); };
    this->m_constraint_solver_timings = this->m_internal_timer.elapsed();
    // clang-format on

    {
      SIMPLE_TRACY_ZONE_SCOPED_N("Simulator::resolveConstraints - apply constraint forces");
      ::pinocchio::evalConstraintJacobianTransposeMatrixProduct(
        this->model(), this->data(),                      //
        this->constraints_problem().constraint_models,    //
        this->constraints_problem().constraint_datas,     //
        this->constraints_problem().constraints_forces(), //
        this->tau_constraints);
      this->tau_total += this->tau_constraints;
    }
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  bool SimulatorTpl<S, O, JointCollectionTpl>::checkCollisionPairs() const
  {
    for (GeomIndex col_pair_id = 0; col_pair_id < this->geom_model().collisionPairs.size(); col_pair_id++)
    {
      const GeomIndex geom_id1 = this->geom_model().collisionPairs[col_pair_id].first;
      const GeomIndex geom_id2 = this->geom_model().collisionPairs[col_pair_id].second;
      const ::pinocchio::GeometryObject & geom1 = this->geom_model().geometryObjects[geom_id1];
      const ::pinocchio::GeometryObject & geom2 = this->geom_model().geometryObjects[geom_id2];
      if (geom1.parentJoint == geom2.parentJoint)
      {
        return false;
      }
    }
    return true;
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  bool SimulatorTpl<S, O, JointCollectionTpl>::check() const
  {
    assert(this->m_model != nullptr && "The model handle points to nullptr.");
    assert(this->m_data != nullptr && "The data handle points to nullptr.");
    assert(this->m_geom_model != nullptr && "The geometry model handle points to nullptr.");
    assert(this->m_geom_data != nullptr && "The geometry data handle points to nullptr.");
    assert(this->m_broad_phase_manager->check() && "The broad phase manager is not aligned with the geometry model.");
    assert(this->vfree.size() == this->model().nv && "The free velocity vector is not of right size.");
    assert(this->vnew.size() == this->model().nv && "The new velocity vector is not of right size.");
    assert(this->ftotal.size() == static_cast<std::size_t>(this->model().njoints) && "The total force vector is not of right size.");
    assert(this->constraints_problem().check() && "The constraint problem is invalid.");
    assert(
      this->checkCollisionPairs()
      && "The GeometryModel contains collision pairs between GeometryObjects attached to the same "
         "joint.");

    return static_cast<bool>(
      this->m_model && this->m_data                                       //
      && this->m_geom_model                                               //
      && this->m_geom_data                                                //
      && this->m_broad_phase_manager->check()                             //
      && this->vfree.size() == this->model().nv                           //
      && this->vnew.size() == this->model().nv                            //
      && this->ftotal.size() == static_cast<std::size_t>(model().njoints) //
      && this->constraints_problem().check()                              //
      && this->checkCollisionPairs());
  }

  // --------------------------------------------------------------------------
  template<typename S, int O, template<typename, int> class JointCollectionTpl>
  void SimulatorTpl<S, O, JointCollectionTpl>::warmStartConstraintForces()
  {
    boost::optional<const Eigen::Ref<const VectorXs>> lambda_guess(this->constraints_problem().constraints_forces());
    // ::pinocchio::computeContactForces(
    //   this->model(), this->data(), this->constraints_problem().constraints_velocities_warmstarts(),
    //   this->constraints_problem().constraint_models, this->constraints_problem().constraint_datas,
    //   this->constraints_problem().contact_compliances(), this->contact_solver_info, lambda_guess);

    const auto problem_size = static_cast<Eigen::Index>(this->constraints_problem().constraints_problem_size());
    this->constraints_problem().constraints_forces() = (this->data().lambda_c.head(problem_size));
  }

  namespace details
  {

    // --------------------------------------------------------------------------
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    void
    SimulatorConstraintSolverTpl<::pinocchio::ADMMContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::setup(Simulator & simulator)
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("SimulatorConstraintSolver<ADMM>::setup");
      ADMMConstraintSolver & solver = simulator.admm_constraint_solver;
      const ADMMConstraintSolverSettings & settings = simulator.admm_constraint_solver_settings;

      // TODO: should we reuse rho and rho_power or always reset them?
      const Scalar mu = settings.mu;
      const Scalar tau = settings.tau;
      const Scalar rho = settings.rho;
      const Scalar rho_power = settings.rho_power;
      const Scalar rho_power_factor = settings.rho_power_factor;
      const Scalar linear_update_rule_factor = settings.linear_update_rule_factor;
      const Scalar ratio_primal_dual = settings.ratio_primal_dual;
      const int lanczos_size = settings.lanczos_size;

      const auto problem_size = static_cast<int>(simulator.constraints_problem().constraints_problem_size());
      if (
        simulator.isReset() || simulator.constraint_solver_type != Simulator::ConstraintSolverType::ADMM
        || solver.getPrimalSolution().size() != problem_size)
      {
        solver = ADMMConstraintSolver(
          problem_size, mu, tau, rho_power, rho_power_factor, //
          linear_update_rule_factor, ratio_primal_dual, lanczos_size);
        solver.setRho(rho);
      }
      else
      {
        solver.setProximalValue(mu);
        solver.setTau(tau);
        solver.setRho(rho);
        solver.setRhoPower(rho_power);
        solver.setRhoPowerFactor(rho_power_factor);
        solver.setLinearUpdateRuleFactor(linear_update_rule_factor);
        solver.setRatioPrimalDual(ratio_primal_dual);
        solver.setLanczosSize(lanczos_size);
      }
      simulator.constraint_solver_type = Simulator::ConstraintSolverType::ADMM;
      solver.setMaxIterations(settings.max_iter);
      solver.setAbsolutePrecision(settings.absolute_precision);
      solver.setRelativePrecision(settings.relative_precision);
    }

    // --------------------------------------------------------------------------
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    void SimulatorConstraintSolverTpl<::pinocchio::ADMMContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::run(
      Simulator & simulator, Scalar dt)
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("SimulatorConstraintSolver<ADMM>::run");
      ConstraintsProblem & constraints_problem = simulator.constraints_problem();
      ADMMConstraintSolver & solver = simulator.admm_constraint_solver;
      const ADMMConstraintSolverSettings & settings = simulator.admm_constraint_solver_settings;

      // Create/update constraint solver
      setup(simulator);
      PINOCCHIO_EIGEN_MALLOC_NOT_ALLOWED();

      // Delassus
      typename Simulator::DelassusOperator G(constraints_problem.constraint_cholesky_decomposition);

      // Drift term
      auto g = constraints_problem.g();

      // Preconditioner
      auto preconditioner = constraints_problem.preconditioner();

      // Input/output of the solver
      auto constraints_forces = constraints_problem.constraints_forces();

      if (!simulator.warm_start_constraints_forces)
      {
        constraints_forces.setZero();
      }

      solver.solve(
        G, g, constraints_problem.constraint_models, dt, boost::make_optional((RefConstVectorXs)preconditioner), //
        boost::make_optional((RefConstVectorXs)constraints_forces), boost::none, constraints_problem.is_ncp, settings.admm_update_rule,
        settings.stat_record);

      // Get solution of the solver
      constraints_forces = solver.getPrimalSolution();
      // Get constraint velocities -> TODO: it's no velocities anymore
      constraints_problem.constraints_velocities() = solver.getDualSolution();
      if (simulator.constraints_problem().is_ncp)
      {
        constraints_problem.constraints_velocities() += -solver.getComplementarityShift();
      }

      // Get time scaling for derivatives
      ::pinocchio::internal::getTimeScalingFromAccelerationToConstraints(
        constraints_problem.constraint_models, dt, constraints_problem.time_scaling_acc_to_constraints());

      PINOCCHIO_EIGEN_MALLOC_ALLOWED();
    }

    // --------------------------------------------------------------------------
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    void SimulatorConstraintSolverTpl<::pinocchio::PGSContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::setup(Simulator & simulator)
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("SimulatorConstraintSolver<PGS>::setup");
      PGSConstraintSolver & solver = simulator.pgs_constraint_solver;
      const PGSConstraintSolverSettings & settings = simulator.pgs_constraint_solver_settings;

      const auto problem_size = static_cast<int>(simulator.constraints_problem().constraints_problem_size());
      if (
        simulator.isReset() || simulator.constraint_solver_type != Simulator::ConstraintSolverType::PGS
        || solver.getPrimalSolution().size() != problem_size)
      {
        solver = PGSConstraintSolver(problem_size);
      }
      simulator.constraint_solver_type = Simulator::ConstraintSolverType::PGS;

      solver.setMaxIterations(settings.max_iter);
      solver.setAbsolutePrecision(settings.absolute_precision);
      solver.setRelativePrecision(settings.relative_precision);
    }

    // --------------------------------------------------------------------------
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    void SimulatorConstraintSolverTpl<::pinocchio::PGSContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::run(
      Simulator & simulator, Scalar dt)
    {
      SIMPLE_TRACY_ZONE_SCOPED_N("SimulatorConstraintSolver<PGS>::run");
      ConstraintsProblem & constraints_problem = simulator.constraints_problem();
      PGSConstraintSolver & solver = simulator.pgs_constraint_solver;
      const PGSConstraintSolverSettings & settings = simulator.pgs_constraint_solver_settings;

      // Create constraint solver
      setup(simulator);

      // Delassus
      typename Simulator::DelassusOperator G(constraints_problem.constraint_cholesky_decomposition);
      const bool enforce_symmetry = true;
      const DelassusOperatorDense delassus_dense(G, enforce_symmetry);

      // Drift term
      auto g = constraints_problem.g();

      // Warm-start and solution of the constraint solver
      auto constraints_forces = constraints_problem.constraints_forces();

      if (!simulator.warm_start_constraints_forces)
      {
        constraints_forces.setZero();
      }

      solver.solve(
        delassus_dense, g, constraints_problem.constraint_models, dt, boost::make_optional((RefConstVectorXs)constraints_forces), //
        settings.over_relax, constraints_problem.is_ncp, settings.stat_record);

      // Get solution of the solver
      constraints_forces = solver.getPrimalSolution();
      // Get constraint velocities
      constraints_problem.constraints_velocities() = solver.getDualSolution();

      // Get time scaling for derivatives
      ::pinocchio::internal::getTimeScalingFromAccelerationToConstraints(
        constraints_problem.constraint_models, dt, constraints_problem.time_scaling_acc_to_constraints());
    }

  } // namespace details

} // namespace simple

#endif // __simple_core_simulator_hxx__
