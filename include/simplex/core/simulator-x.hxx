#ifndef __simplex_core_simulator_x_hxx__
#define __simplex_core_simulator_x_hxx__

#include "simplex/core/simulator-x.hpp"

namespace simplex
{
    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorXTpl(
        ModelHandle model_handle,              //
        DataHandle data_handle,                //
        GeometryModelHandle geom_model_handle, //
        GeometryDataHandle geom_data_handle)
    : model_(model_handle)
    , data_(data_handle)
    , geom_model_(geom_model_handle)
    , geom_data_(geom_data_handle)
    {
        init();
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorXTpl(ModelHandle model_handle, GeometryModelHandle geom_model_handle)
    : SimulatorXTpl(
          model_handle,
          std::make_shared<::pinocchio::Data>(*model_handle),
          geom_model_handle,
          std::make_shared<::pinocchio::GeometryData>(*geom_model_handle))
    {
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorState::init(int nq, int nv, int njoints)
    {
        PINOCCHIO_CHECK_INPUT_ARGUMENT(nq >= 0, "nq is negative");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(nv >= 0, "nv is negative");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(nq >= nv, "nq should be bigger than nv");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(njoints >= 0, "njoints is negative");

        q.resize(nq);
        v.resize(nv);
        tau.resize(nv);
        fext.resize(static_cast<std::size_t>(njoints), Force::Zero());
        dt = Scalar(-1);
        constraint_solver_type = ConstraintSolverType::NONE;
        tau_damping.resize(nv);
        qnew.resize(nq);
        vfree.resize(nv);
        afree.resize(nv);
        vnew.resize(nv);
        anew.resize(nv);
        tau_total.resize(nv);
        tau_constraints.resize(nv);
        is_reset = true;
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorState::reset()
    {
        // We assume model/geom_model have not changed, so we don't need to resize the state vectors.
        dt = Scalar(-1);
        constraint_solver_type = ConstraintSolverType::NONE;
        is_reset = true;
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorWorkspace::init(
        ModelHandle model, DataHandle data, GeometryModelHandle geom_model, GeometryDataHandle geom_data)
    {
        using ::pinocchio::helper::get_ref;

        PINOCCHIO_CHECK_INPUT_ARGUMENT(model != nullptr, "model is nullptr");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(data != nullptr, "data is nullptr");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(geom_model != nullptr, "geom_model is nullptr");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(geom_data != nullptr, "geom_data is nullptr");

        // Setup broad phase
        for (::pinocchio::GeometryObject & geom : get_ref(geom_model).geometryObjects)
        {
            geom.geometry->computeLocalAABB();
        }
        collision_callback_ = std::make_shared<CollisionCallBackCollect>(get_ref(geom_model), get_ref(geom_data));
        broadphase_manager_ = std::make_shared<BroadPhaseManager>(&get_ref(model), &get_ref(geom_model), &get_ref(geom_data));

        // Setup narrow phase
        for (coal::CollisionRequest & request : get_ref(geom_data).collisionRequests)
        {
            request.enable_contact = true;
        }

        // Setup constraint problem
        // TODO: this wipes all the constraints in the constraint problem, including point/frame anchors.
        // Should we do that?
        constraint_problem_ = std::make_shared<ConstraintsProblemDerivatives>(model, data, geom_model, geom_data);

        // Setup constraint solvers
        constraint_solvers.admm_solver = ADMMConstraintSolver(0);
        constraint_solvers.pgs_solver = PGSConstraintSolver(0);
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
        constraint_solvers.clarabel_solver = ClarabelConstraintSolver(0);
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

        // Temporary variables used for computation
        vnew_integration_tmp.resize(get_ref(model).nv);
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorWorkspace::reset()
    {
        constraint_problem().clear();
        constraint_solvers.pgs_solver.reset();
        constraint_solvers.admm_solver.reset();
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
        constraint_solvers.clarabel_solver.reset();
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorTimings::clear()
    {
        timings_broadphase_collision_detection.clear();
        timings_narrowphase_collision_detection.clear();
        timings_collision_detection.clear();
        timings_constraint_solver.clear();
        timings_step.clear();

        timer_step.stop();
        timer_internal.stop();
        timer_collision_detection.stop();
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::addPointAnchorConstraints(
        const BilateralPointConstraintModelVector & point_anchor_constraint_models)
    {
        auto & workspace_bmodels = workspace.constraint_problem().bilateral_point_constraint_models;
        auto & workspace_bdatas = workspace.constraint_problem().bilateral_point_constraint_datas;
        for (const auto & bmodel : point_anchor_constraint_models)
        {
            workspace_bmodels.emplace_back(bmodel);
            workspace_bdatas.emplace_back(workspace_bmodels.back().createData());
        }

        int current_workspace_max_size = int(workspace.constraint_problem().storage.g_storage.capacity());
        int point_anchor_size = workspace.constraint_problem().bilateral_constraints_size();
        workspace.constraint_problem().storage.reserve(current_workspace_max_size + point_anchor_size);
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void
    SimulatorXTpl<S, O, JointCollectionTpl>::addFrameAnchorConstraints(const WeldConstraintModelVector & frame_anchor_constraint_models)
    {
        auto & workspace_wmodels = workspace.constraint_problem().weld_constraint_models;
        auto & workspace_wdatas = workspace.constraint_problem().weld_constraint_datas;
        for (const auto & wmodel : frame_anchor_constraint_models)
        {
            workspace_wmodels.emplace_back(wmodel);
            workspace_wdatas.emplace_back(workspace_wmodels.back().createData());
        }

        int current_workspace_max_size = int(workspace.constraint_problem().storage.g_storage.capacity());
        int frame_anchor_size = workspace.constraint_problem().weld_constraints_size();
        workspace.constraint_problem().storage.reserve(current_workspace_max_size + frame_anchor_size);
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::reset()
    {
        state.reset();
        workspace.reset();
        timings.clear();
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::init()
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::init");
        state.init(model().nq, model().nv, model().njoints);
        workspace.init(model_, data_, geom_model_, geom_data_);
        reset();
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<template<typename> class ConstraintSolver, typename ConfigVectorType, typename VelocityVectorType, typename TorqueVectorType>
    void SimulatorXTpl<S, O, JointCollectionTpl>::step(
        const Eigen::MatrixBase<ConfigVectorType> & q,
        const Eigen::MatrixBase<VelocityVectorType> & v,
        const Eigen::MatrixBase<TorqueVectorType> & tau,
        Scalar dt)
    {
        state.fext.assign((std::size_t)(model().njoints), Force::Zero());
        step<ConstraintSolver>(q, v, tau, state.fext, dt);
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<
        template<typename> class ConstraintSolver,
        typename ConfigVectorType,
        typename VelocityVectorType,
        typename TorqueVectorType,
        typename ForceDerived>
    void SimulatorXTpl<S, O, JointCollectionTpl>::step(
        const Eigen::MatrixBase<ConfigVectorType> & q,
        const Eigen::MatrixBase<VelocityVectorType> & v,
        const Eigen::MatrixBase<TorqueVectorType> & tau,
        const ::pinocchio::container::aligned_vector<ForceDerived> & fext,
        const Scalar dt)
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::step");
        if (config.measure_timings)
        {
            timings.timer_step.start();
        }

        // TODO(louis): should we use check arguments or assert/throw?
        PINOCCHIO_CHECK_ARGUMENT_SIZE(
            state.vfree.size(), model().nv,
            "The sizes of the free velocity of the simulator and the input velocity do not match. "
            "You problably changed your model, data, geom_model or geom_data and forgot to call "
            "allocate().");
        PINOCCHIO_CHECK_ARGUMENT_SIZE(q.size(), model().nq, "The joint configuration vector is not of right size");
        PINOCCHIO_CHECK_ARGUMENT_SIZE(v.size(), model().nv, "The joint velocity vector is not of right size");
        PINOCCHIO_CHECK_ARGUMENT_SIZE(tau.size(), model().nv, "The joint torque vector is not of right size");
        PINOCCHIO_CHECK_ARGUMENT_SIZE(
            fext.size(), static_cast<std::size_t>(model().njoints), "The external forces vector is not of right size");
        PINOCCHIO_CHECK_INPUT_ARGUMENT(dt >= Scalar(0), "dt is not >= 0");

        // Record state of simulator
        state.q = q;
        state.v = v;
        state.tau = tau;
        state.tau_damping = -model().damping.cwiseProduct(v);
        state.fext = fext;
        state.dt = dt;
        assert(check(false) && "The simulator is not properly instanciated.");

        // Set up data for downstream algorithms
        data().q_in = q;
        data().v_in = v;
        data().tau_in = tau;

        // Compute the mass matrix of the system - used by the delassus cholesky operator
        if (workspace.constraint_problem().delassus_type == DelassusType::CHOLESKY
            || workspace.constraint_problem().delassus_type == DelassusType::DENSE)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::step - compute CRBA");
            ::pinocchio::crba(model(), data(), state.q, pinocchio::Convention::WORLD);
        }

        // Compute free acceleration of the system
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::step - compute vfree (first call to ABA)");
            state.tau_total = state.tau + state.tau_damping;
            state.afree = ::pinocchio::aba(model(), data(), state.q, state.v, state.tau_total, state.fext, pinocchio::Convention::WORLD);
            state.vfree = state.v + state.dt * state.afree;
        }

        // Collision detection
        detectCollisions();

        // Update constraint problem with result of collision detection
        workspace.constraint_problem().update(state.dt);
        assert(check(true) && "The simulator and/or constraints are not properly instanciated.");

        state.tau_constraints.setZero();
        if (workspace.constraint_problem().constraints_problem_size() > 0)
        {
            // Constraint resolution - compute the total joint torques (if there are any collisions)
            /// i.e. input system torques + constraint torques
            resolveConstraints<ConstraintSolver>();

            {
                SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::step - compute vnew (second call to ABA)");
                state.anew = ::pinocchio::aba(model(), data(), state.q, state.v, state.tau_total, state.fext, pinocchio::Convention::WORLD);
                state.vnew = state.v + state.dt * state.anew;
            }
        }
        else
        {
            // Reset internal quantities of constraint solvers (num its, decomposition count, stats etc)
            workspace.constraint_solvers.pgs_solver.reset();
            workspace.constraint_solvers.admm_solver.reset();
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
            workspace.constraint_solvers.clarabel_solver.reset();
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

            state.anew = (state.vfree - state.v) / state.dt; // TODO(quentin) do it differrently
            state.vnew = state.vfree;

            // Reset constraint solver timings
            timings.timings_constraint_solver.clear();
        }

        workspace.vnew_integration_tmp = state.vnew * state.dt;
        ::pinocchio::integrate(model(), state.q, workspace.vnew_integration_tmp, state.qnew);
        state.is_reset = false;

        if (config.measure_timings)
            timings.timer_step.stop();
        timings.timings_step = timings.timer_step.elapsed();
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void SimulatorXTpl<S, O, JointCollectionTpl>::detectCollisions()
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::detectCollisions");

        if (config.measure_timings)
            timings.timer_internal.start();

        // Compute oMg for each geometry
        ::pinocchio::updateGeometryPlacements(model(), data(), geom_model(), geom_data());

        // Reset collision results - super important! Otherwise constraint from previous time step may be
        // detected between non-colliding geometries.
        for (coal::CollisionResult & col_res : geom_data().collisionResults)
        {
            col_res.clear();
        }

        // Run broad + narrow phase collision detection
        if (config.measure_timings)
            timings.timer_collision_detection.start();

        const bool recompute_local_aabb = false; // already computed in `SimulatorWorkspace::init`
        workspace.broadphase_manager().update(recompute_local_aabb);
        assert(workspace.broadphase_manager().check() && "The broad phase manager is not aligned with the geometry model.");

        // --> Broad Phase
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::detectCollisions - Broad phase");
            ::pinocchio::computeCollisions(workspace.broadphase_manager(), &workspace.collision_callback());
        }

        if (config.measure_timings)
            timings.timer_collision_detection.stop();
        timings.timings_broadphase_collision_detection = timings.timer_collision_detection.elapsed();

        // --> Narrow Phase
        if (config.measure_timings)
            timings.timer_collision_detection.start();
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::detectCollisions - Narrow phase");
            for (std::size_t i = 0; i < workspace.collision_callback().pair_indexes.size(); ++i)
            {
                const std::size_t pair_index = workspace.collision_callback().pair_indexes[i];
                const ::pinocchio::CollisionPair & cp = geom_model().collisionPairs[pair_index];
                // const ::pinocchio::GeometryObject & obj1 = geom_model().geometryObjects[cp.first];
                // const ::pinocchio::GeometryObject & obj2 = geom_model().geometryObjects[cp.second];

                // middle phase collision detection
                // TODO(louis): why is this bugged?
                coal::CollisionRequest collision_request(geom_data().collisionRequests[pair_index]);
                bool obb_overlap = true;
                // TODO: fix middle phase!
                // if (
                //   obj1.geometry->getNodeType() == coal::GEOM_PLANE || obj1.geometry->getNodeType() == coal::GEOM_HALFSPACE
                //   || obj2.geometry->getNodeType() == coal::GEOM_PLANE || obj2.geometry->getNodeType() == coal::GEOM_HALFSPACE)
                // {
                //   obb_overlap = true;
                // }
                // else
                // {
                //   coal::Transform3s oM1(toFclTransform3f(geom_data().oMg[cp.first]));
                //   coal::Transform3s oM2(toFclTransform3f(geom_data().oMg[cp.second]));
                //   const Scalar security_margin = collision_request.security_margin;
                //   //
                //   const coal::AABB aabb1 = obj1.geometry->aabb_local.expand(security_margin * 0.5);
                //   coal::OBB obb1;
                //   coal::convertBV(aabb1, oM1, obb1);
                //   //
                //   const coal::AABB aabb2 = obj1.geometry->aabb_local.expand(security_margin * 0.5);
                //   coal::OBB obb2;
                //   coal::convertBV(aabb2, oM2, obb2);
                //   obb_overlap = obb1.overlap(obb2);
                // }
                try
                {
                    if (obb_overlap)
                    {
                        pinocchio::computeCollision(geom_model(), geom_data(), pair_index, collision_request);
                        ::pinocchio::computeContactPatch(geom_model(), geom_data(), pair_index);
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
        if (config.measure_timings)
            timings.timer_collision_detection.stop();
        timings.timings_narrowphase_collision_detection = timings.timer_collision_detection.elapsed();

        if (config.measure_timings)
            timings.timer_internal.stop();
        timings.timings_collision_detection = timings.timer_internal.elapsed();
    }

    namespace details
    {
        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        void
        SimulatorXConstraintSolverTpl<::simplex::ADMMContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::run(SimulatorX & simulator)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorXConstraintSolver<ADMM>::run");
            ConstraintsProblemDerivatives & constraint_problem = simulator.workspace.constraint_problem();
            ADMMConstraintSolver & admm_solver = simulator.workspace.constraint_solvers.admm_solver;
            const ADMMConstraintSolverConfig & admm_config = simulator.config.constraint_solvers_configs.admm_config;

            // Create/update constraint solver
            setup(simulator);
            PINOCCHIO_EIGEN_MALLOC_NOT_ALLOWED();

            // Input/output of the solver
            // auto impulses = constraint_problem.constraint_forces();
            // impulses.array() *= dt;
            // boost::optional<RefConstVectorXs> primal_guess(impulses);
            // TODO: now we only consider contact impulses, in the future we may want to warmstart all constraints
            boost::optional<RefConstVectorXs> primal_guess(boost::none);
            auto constraint_velocities = constraint_problem.constraint_velocities;
            boost::optional<RefConstVectorXs> dual_guess(constraint_velocities);
            boost::optional<RefConstVectorXs> preconditioner = boost::none;
            boost::optional<Scalar> mu_prox0(admm_config.mu_prox_prev);
            boost::optional<Scalar> rho0(admm_config.rho_prev);

            // Warm starts
            if (!simulator.config.warmstart_constraint_velocities)
            {
                dual_guess = boost::none;
            }
            const bool temp_is_ok = simulator.isReset() || !admm_solver.isInitialized();
            if (!admm_config.warmstart_mu_prox //
                || temp_is_ok                  // if solver is not initialized, then solver is new and has never run before
            )
            {
                mu_prox0 = admm_config.mu_prox;
            }
            if (!admm_config.warmstart_rho //
                || temp_is_ok              //
            )
            {
                rho0 = boost::none; // use lanczos to compute initial rho
            }

            // Drift term
            auto g = constraint_problem.g;

#define SOLVE_CONSTRAINT_PROBLEM                                                                                                           \
    admm_solver.solve(                                                                                                                     \
        delassus, g, constraint_problem.constraint_models, simulator.state.dt, preconditioner, primal_guess, dual_guess,                   \
        constraint_problem.is_ncp, admm_config.admm_update_rule, rho0, mu_prox0, admm_config.stat_record);

            switch (constraint_problem.delassus_type)
            {
            case (DelassusType::DENSE): {
                // TODO: remove malloc
                typename SimulatorX::DelassusCholeskyExpressionOperator delassus_cholesky(
                    constraint_problem.constraint_cholesky_decomposition);
                typename SimulatorX::DelassusDenseOperator delassus(delassus_cholesky);
                SOLVE_CONSTRAINT_PROBLEM;
                break;
            }
            case (DelassusType::CHOLESKY): {
                typename SimulatorX::DelassusCholeskyExpressionOperator delassus(constraint_problem.constraint_cholesky_decomposition);
                SOLVE_CONSTRAINT_PROBLEM;
                break;
            }
            case (DelassusType::RIGID_BODY): {
                typename SimulatorX::DelassusRigidBodyOperator delassus(
                    pinocchio::helper::make_ref(simulator.model()),                    //
                    pinocchio::helper::make_ref(simulator.data()),                     //
                    pinocchio::helper::make_ref(constraint_problem.constraint_models), //
                    pinocchio::helper::make_ref(constraint_problem.constraint_datas));
                // TODO: constructor m_solve_in_place, m_apply_on_the_right
                // TODO: resize in update
                // TODO: read compliance from constraint models in update
                delassus.compute(true, true);
                SOLVE_CONSTRAINT_PROBLEM;
                break;
            }
            }
#undef SOLVE_CONSTRAINT_PROBLEM

            // Get solution of the solver
            constraint_problem.constraint_forces.array() = admm_solver.getPrimalSolution().array() / simulator.state.dt;
            // Get constraint velocities
            constraint_problem.constraint_velocities = admm_solver.getDualSolution();
            if (constraint_problem.is_ncp)
            {
                constraint_problem.constraint_velocities -= admm_solver.getComplementarityShift();
            }

            // Store rho and mu_prox
            // REFACTOR: this should be stored in solver state
            admm_config.rho_prev = admm_solver.getRho();
            admm_config.mu_prox_prev = admm_solver.getProximalValue();

            // Get time scaling for derivatives - TODO: this is no longer needed!
            ::pinocchio::internal::getTimeScalingFromAccelerationToConstraints(
                constraint_problem.constraint_models, //
                simulator.state.dt, constraint_problem.time_scaling_acc_to_constraints);

            PINOCCHIO_EIGEN_MALLOC_ALLOWED();
        }

        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        void
        SimulatorXConstraintSolverTpl<::simplex::ADMMContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::setup(SimulatorX & simulator)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorXConstraintSolver<ADMM>::setup");
            ADMMConstraintSolver & admm_solver = simulator.workspace.constraint_solvers.admm_solver;
            const ADMMConstraintSolverConfig & admm_config = simulator.config.constraint_solvers_configs.admm_config;

            // Note: we don't set rho here, it is an input of the solver.
            const Scalar tau_prox = admm_config.tau_prox;
            const Scalar tau = admm_config.tau;
            const Scalar rho_power = admm_config.rho_power;
            const Scalar rho_power_factor = admm_config.rho_power_factor;
            const Scalar linear_update_rule_factor = admm_config.linear_update_rule_factor;
            const Scalar ratio_primal_dual = admm_config.ratio_primal_dual;
            const int lanczos_size = admm_config.lanczos_size;
            const int max_delassus_decomposition_updates = admm_config.max_delassus_decomposition_updates;
            const Scalar dual_momentum = admm_config.dual_momentum;
            const Scalar rho_momentum = admm_config.rho_momentum;
            const Scalar rho_update_ratio = admm_config.rho_update_ratio;
            const int rho_min_update_frequency = admm_config.rho_min_update_frequency;

            const auto problem_size = static_cast<int>(simulator.workspace.constraint_problem().constraints_problem_size());
            if (simulator.isReset() || simulator.state.constraint_solver_type != SimulatorState::ConstraintSolverType::ADMM
                || admm_solver.getPrimalSolution().size() != problem_size)
            {
                admm_solver = ADMMConstraintSolver(
                    problem_size, tau_prox, tau, rho_power, rho_power_factor,        //
                    linear_update_rule_factor, ratio_primal_dual, lanczos_size,      //
                    max_delassus_decomposition_updates, dual_momentum, rho_momentum, //
                    rho_update_ratio, rho_min_update_frequency);
            }
            else
            {
                admm_solver.setProximalValue(tau_prox);
                admm_solver.setTau(tau);
                admm_solver.setRhoPower(rho_power);
                admm_solver.setRhoPowerFactor(rho_power_factor);
                admm_solver.setLinearUpdateRuleFactor(linear_update_rule_factor);
                admm_solver.setRatioPrimalDual(ratio_primal_dual);
                admm_solver.setLanczosSize(lanczos_size);
                admm_solver.setMaxDelassusDecompositionUpdates(max_delassus_decomposition_updates);
                admm_solver.setDualMomentum(dual_momentum);
                admm_solver.setRhoMomentum(rho_momentum);
                admm_solver.setRhoUpdateRatio(rho_update_ratio);
                admm_solver.setRhoMinUpdateFrequency(rho_min_update_frequency);
            }
            simulator.state.constraint_solver_type = SimulatorState::ConstraintSolverType::ADMM;
            admm_solver.setMaxIterations(admm_config.max_iter);
            admm_solver.setAbsolutePrecision(admm_config.absolute_precision);
            admm_solver.setRelativePrecision(admm_config.relative_precision);
        }

        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        void
        SimulatorXConstraintSolverTpl<::pinocchio::PGSContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::run(SimulatorX & simulator)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorXConstraintSolver<PGS>::run");
            ConstraintsProblemDerivatives & constraint_problem = simulator.workspace.constraint_problem();
            PGSConstraintSolver & pgs_solver = simulator.workspace.constraint_solvers.pgs_solver;
            const PGSConstraintSolverConfig & pgs_config = simulator.config.constraint_solvers_configs.pgs_config;

            // Create constraint solver
            setup(simulator);

            // Warm-start and solution of the constraint solver
            auto impulses = constraint_problem.constraint_forces;
            // impulses.array() *= simulator.state.dt;

            // TODO(louis): warmstart PGS using constraint velocities warmstart + small rho in delassus
            impulses.setZero();
            boost::optional<RefConstVectorXs> primal_guess(impulses);

            // Drift term
            auto g = constraint_problem.g;

            // Delassus
            const bool enforce_symmetry = true;
            auto G = constraint_problem.delassus_matrix;
            // PGS needs to work with an invertible delassus matrix.
            // We use updateDamping to add a rho * Identity diagonal term to the delassus.
            // TODO(louis): make this rho a parameter
            const Scalar rho = 1e-6;
            switch (constraint_problem.delassus_type)
            {
            case (DelassusType::DENSE):
            case (DelassusType::CHOLESKY): {
                typename SimulatorX::DelassusCholeskyExpressionOperator delassus(constraint_problem.constraint_cholesky_decomposition);
                delassus.updateDamping(rho);
                delassus.matrix(G, enforce_symmetry);
                break;
            }
            case (DelassusType::RIGID_BODY): {
                typename SimulatorX::DelassusRigidBodyOperator delassus(
                    pinocchio::helper::make_ref(simulator.model()),                    //
                    pinocchio::helper::make_ref(simulator.data()),                     //
                    pinocchio::helper::make_ref(constraint_problem.constraint_models), //
                    pinocchio::helper::make_ref(constraint_problem.constraint_datas));
                delassus.updateDamping(rho);
                delassus.compute(true, true);
                delassus.matrix(G, enforce_symmetry);
                break;
            }
            }

            const DelassusOperatorDense delassus_op(G);

            // Run constraint solver
            pgs_solver.solve(
                delassus_op, g, //
                constraint_problem.constraint_models, simulator.state.dt, primal_guess, pgs_config.over_relax, constraint_problem.is_ncp,
                pgs_config.stat_record);

            // Get solution of the solver
            constraint_problem.constraint_forces.array() = pgs_solver.getPrimalSolution().array() / simulator.state.dt;
            // Get constraint velocities
            constraint_problem.constraint_velocities = pgs_solver.getDualSolution();

            // Get time scaling for derivatives
            ::pinocchio::internal::getTimeScalingFromAccelerationToConstraints(
                constraint_problem.constraint_models, //
                simulator.state.dt, constraint_problem.time_scaling_acc_to_constraints);
        }

        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        void SimulatorXConstraintSolverTpl<::pinocchio::PGSContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::setup(
            SimulatorX & simulator)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorXConstraintSolver<PGS>::setup");
            PGSConstraintSolver & pgs_solver = simulator.workspace.constraint_solvers.pgs_solver;
            const PGSConstraintSolverConfig & pgs_config = simulator.config.constraint_solvers_configs.pgs_config;

            const auto problem_size = static_cast<int>(simulator.workspace.constraint_problem().constraints_problem_size());
            if (simulator.isReset() || simulator.state.constraint_solver_type != SimulatorState::ConstraintSolverType::PGS
                || pgs_solver.getPrimalSolution().size() != problem_size)
            {
                pgs_solver = PGSConstraintSolver(problem_size);
            }
            simulator.state.constraint_solver_type = SimulatorState::ConstraintSolverType::PGS;

            pgs_solver.setMaxIterations(pgs_config.max_iter);
            pgs_solver.setAbsolutePrecision(pgs_config.absolute_precision);
            pgs_solver.setRelativePrecision(pgs_config.relative_precision);
        }

#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        void SimulatorXConstraintSolverTpl<::simplex::ClarabelContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::run(
            SimulatorX & simulator)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorConstraintSolver<Clarabel>::run");
            ConstraintsProblemDerivatives & constraint_problem = simulator.workspace.constraint_problem();
            ClarabelConstraintSolver & clarabel_solver = simulator.workspace.constraint_solvers.clarabel_solver;
            const ClarabelConstraintSolverConfig & clarabel_config = simulator.config.constraint_solvers_configs.clarabel_config;

            // Create/update constraint solver
            setup(simulator);

            // Input/output of the solver
            boost::optional<RefConstVectorXs> primal_guess(boost::none);
            auto constraint_velocities = constraint_problem.constraint_velocities;
            boost::optional<RefConstVectorXs> dual_guess(
                simulator.config.warmstart_constraint_velocities ? boost::optional<RefConstVectorXs>(constraint_velocities) : boost::none);
            boost::optional<RefConstVectorXs> preconditioner = boost::none;

            // Drift term
            auto g = constraint_problem.g;

            // Solve constraint problem using Clarabel
            switch (constraint_problem.delassus_type)
            {
            case (DelassusType::DENSE):
            case (DelassusType::CHOLESKY): {
                typename SimulatorX::DelassusCholeskyExpressionOperator delassus(constraint_problem.constraint_cholesky_decomposition);
                clarabel_solver.solve(
                    delassus, g, constraint_problem.constraint_models, constraint_problem.constraint_datas, preconditioner, primal_guess,
                    dual_guess, constraint_problem.is_ncp, clarabel_config.stat_record);
                break;
            }
            case (DelassusType::RIGID_BODY): {
                typename SimulatorX::DelassusRigidBodyOperator delassus(
                    pinocchio::helper::make_ref(simulator.model()),                    //
                    pinocchio::helper::make_ref(simulator.data()),                     //
                    pinocchio::helper::make_ref(constraint_problem.constraint_models), //
                    pinocchio::helper::make_ref(constraint_problem.constraint_datas));
                delassus.compute(true, true);
                clarabel_solver.solve(
                    delassus, g, constraint_problem.constraint_models, constraint_problem.constraint_datas, preconditioner, primal_guess,
                    dual_guess, constraint_problem.is_ncp, clarabel_config.stat_record);
                break;
            }
            }

            // Get solution of the solver
            constraint_problem.constraint_forces.array() = clarabel_solver.getPrimalSolution().array() / simulator.state.dt;
            // Get constraint velocities
            constraint_problem.constraint_velocities = clarabel_solver.getDualSolution();

            // Get time scaling for derivatives
            ::pinocchio::internal::getTimeScalingFromAccelerationToConstraints(
                constraint_problem.constraint_models, simulator.state.dt, constraint_problem.time_scaling_acc_to_constraints);
        }

        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        void SimulatorXConstraintSolverTpl<::simplex::ClarabelContactSolverTpl, _Scalar, _Options, JointCollectionTpl>::setup(
            SimulatorX & simulator)
        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorXConstraintSolver<Clarabel>::setup");
            ClarabelConstraintSolver & clarabel_solver = simulator.workspace.constraint_solvers.clarabel_solver;
            const ClarabelConstraintSolverConfig & clarabel_config = simulator.config.constraint_solvers_configs.clarabel_config;

            const auto problem_size = static_cast<int>(simulator.workspace.constraint_problem().constraints_problem_size());
            if (simulator.isReset() || simulator.state.constraint_solver_type != SimulatorState::ConstraintSolverType::CLARABEL
                || clarabel_solver.getPrimalSolution().size() != problem_size)
            {
                clarabel_solver = ClarabelConstraintSolver(problem_size);
            }
            simulator.state.constraint_solver_type = SimulatorState::ConstraintSolverType::CLARABEL;

            clarabel_solver.setMaxIterations(clarabel_config.max_iter);
            clarabel_solver.setAbsolutePrecision(clarabel_config.absolute_precision);
            clarabel_solver.setRelativePrecision(clarabel_config.relative_precision);
        }
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

    } // namespace details

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<template<typename> class ConstraintSolver>
    void SimulatorXTpl<S, O, JointCollectionTpl>::resolveConstraints()
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::resolveConstraints");

        assert(
            !workspace.constraint_problem().constraint_models.empty() //
            && "Resolve collisions should not be called if there are no constraints.");

        // Build the constraint quantities for the constraint solver
        // TODO(quentin): warm-start constraint forces via constraint inverse dynamics
        workspace.constraint_problem().build(state.vfree, state.v, state.dt);
        preambleResolveConstraints(); // now unused

        // Call the constraint solver
        if (config.measure_timings)
            timings.timer_internal.start();
        details::SimulatorXConstraintSolverTpl<ConstraintSolver, S, O, JointCollectionTpl>::run(*this);
        if (config.measure_timings)
            timings.timer_internal.stop();
        timings.timings_constraint_solver = timings.timer_internal.elapsed();

        {
            SIMPLEX_TRACY_ZONE_SCOPED_N("SimulatorX::resolveConstraints - apply constraint forces");
            ::pinocchio::evalConstraintJacobianTransposeMatrixProduct(
                model(), data(),                                  //
                workspace.constraint_problem().constraint_models, //
                workspace.constraint_problem().constraint_datas,  //
                workspace.constraint_problem().constraint_forces, //
                state.tau_constraints);
            state.tau_total += state.tau_constraints;
        }
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    bool SimulatorXTpl<S, O, JointCollectionTpl>::checkCollisionPairs() const
    {
        for (GeomIndex col_pair_id = 0; col_pair_id < geom_model().collisionPairs.size(); col_pair_id++)
        {
            const GeomIndex geom_id1 = geom_model().collisionPairs[col_pair_id].first;
            const GeomIndex geom_id2 = geom_model().collisionPairs[col_pair_id].second;
            const ::pinocchio::GeometryObject & geom1 = geom_model().geometryObjects[geom_id1];
            const ::pinocchio::GeometryObject & geom2 = geom_model().geometryObjects[geom_id2];
            if (geom1.parentJoint == geom2.parentJoint)
            {
                return false;
            }
        }
        return true;
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    bool SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorState::check(const Model & model) const
    {
        PINOCCHIO_UNUSED_VARIABLE(model);
        assert(q.size() == model.nq && "The configuration vector `q` is not of right size.");
        assert(v.size() == model.nv && "The velocity vector `v` is not of right size.");
        assert(tau.size() == model.nv && "The torque vector `tau` is not of right size.");
        assert(dt > 0 && "dt should be > 0");
        assert(fext.size() == static_cast<std::size_t>(model.njoints) && "The external force vector `fext` is not of right size.");
        assert(tau_damping.size() == model.nv && "The joint damping vector `tau_damping` is not of right size.");
        assert(qnew.size() == model.nq && "The configuration vector `qnew` is not of right size.");
        assert(vfree.size() == model.nv && "The free velocity vector `vfree` is not of right size.");
        assert(afree.size() == model.nv && "The free acceleration vector `afree` is not of right size.");
        assert(vnew.size() == model.nv && "The new velocity vector `vnew` is not of right size.");
        assert(anew.size() == model.nv && "The new acceleration vector `anew` is not of right size.");
        assert(tau_total.size() == model.nv && "The total torque vector `tau_total` is not of right size.");
        assert(tau_constraints.size() == model.nv && "The constraints torque vector `tau_constraints` is not of right size.");

        return true;
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    bool SimulatorXTpl<S, O, JointCollectionTpl>::SimulatorWorkspace::check(
        const Model & model, const bool constraint_problem_has_been_updated) const
    {
        PINOCCHIO_UNUSED_VARIABLE(model);
        if (constraint_problem_has_been_updated)
        {
            assert(constraint_problem().check() && "The constraint problem is invalid.");
        }
        assert(broadphase_manager().check() && "The broad phase manager is not aligned with the geometry model.");
        assert(vnew_integration_tmp.size() == model.nv && "Temporary velocity vector is not of right size.");

        return true;
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    bool SimulatorXTpl<S, O, JointCollectionTpl>::check(const bool constraint_problem_has_been_updated) const
    {
        PINOCCHIO_UNUSED_VARIABLE(constraint_problem_has_been_updated);
        assert(model_ != nullptr && "The model handle points to nullptr.");
        assert(data_ != nullptr && "The data handle points to nullptr.");
        assert(geom_model_ != nullptr && "The geometry model handle points to nullptr.");
        assert(geom_data_ != nullptr && "The geometry data handle points to nullptr.");
        assert(workspace.check(model(), constraint_problem_has_been_updated) && "Workspace check has failed.");
        assert(state.check(model()) && "State check has failed.");
        assert(
            checkCollisionPairs()
            && "The GeometryModel contains collision pairs between GeometryObjects attached to the same "
               "joint.");

        return true;
    }

} // namespace simplex

#endif // ifndef __simplex_core_simulator_x_hxx__