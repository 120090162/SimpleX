#ifndef __simplex_python_core_simulator_x_hpp__
#define __simplex_python_core_simulator_x_hpp__

#include "simplex/bindings/python/fwd.hpp"
#include "simplex/core/simulator-x.hpp"

#include <pinocchio/bindings/python/utils/std-vector.hpp>
#include <pinocchio/bindings/python/utils/copyable.hpp>

namespace simplex
{
    namespace python
    {
        namespace bp = boost::python;

        template<typename SimulatorX>
        struct SimulatorXPythonVisitor : public boost::python::def_visitor<SimulatorXPythonVisitor<SimulatorX>>
        {
            using Self = SimulatorX;
            using Scalar = typename Self::Scalar;

            using Model = typename Self::Model;
            using ModelHandle = typename Self::ModelHandle;
            using Data = typename Self::Data;
            using DataHandle = typename Self::DataHandle;
            using GeometryModel = typename Self::GeometryModel;
            using GeometryModelHandle = typename Self::GeometryModelHandle;
            using GeometryData = typename Self::GeometryData;
            using GeometryDataHandle = typename Self::GeometryDataHandle;

            using CollisionCallBackCollect = typename Self::CollisionCallBackCollect;
            using BroadPhaseManager = typename Self::BroadPhaseManager;
            using ConstraintsProblemDerivatives = typename Self::ConstraintsProblemDerivatives;
            using BilateralPointConstraintModelVector = typename Self::BilateralPointConstraintModelVector;
            using WeldConstraintModelVector = typename Self::WeldConstraintModelVector;

            using VectorXs = typename Self::VectorXs;
            using MatrixXs = typename Self::MatrixXs;
            using SpatialForce = typename Self::SpatialForce;
            using SpatialForceVector = typename Self::SpatialForceVector;

            using SimulatorConfig = typename Self::SimulatorConfig;
            using ConstraintSolverConfigContainer = typename SimulatorConfig::ConstraintSolverConfigContainer;

            using SimulatorState = typename Self::SimulatorState;
            using ConstraintSolverType = typename SimulatorState::ConstraintSolverType;
            using SimulatorWorkspace = typename Self::SimulatorWorkspace;
            using ConstraintSolverContainer = typename SimulatorWorkspace::ConstraintSolverContainer;
            using SimulatorTimings = typename Self::SimulatorTimings;

            static void step_wrapper(
                Self & self,
                const VectorXs & q,
                const VectorXs & v,
                const VectorXs & tau,
                const Scalar dt,
                const ConstraintSolverType solver_type,
                std::size_t nsteps)
            {
                // clang-format off
        #define SIMPLEX_STEP_LOOP(solver_pinocchio_type)                                           \
          self.template step<solver_pinocchio_type>(q, v, tau, dt);                               \
          for (std::size_t i = 0; i < nsteps - 1; ++i)                                            \
          {                                                                                       \
            self.template step<solver_pinocchio_type>(self.state.qnew, self.state.vnew, tau, dt); \
          }
                // clang-format on

                switch (solver_type)
                {
                case ConstraintSolverType::ADMM:
                    SIMPLEX_STEP_LOOP(::simplex::ADMMContactSolverTpl);
                    break;
                case ConstraintSolverType::PGS:
                    SIMPLEX_STEP_LOOP(::pinocchio::PGSContactSolverTpl);
                    break;
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                case ConstraintSolverType::CLARABEL:
                    SIMPLEX_STEP_LOOP(::simplex::ClarabelContactSolverTpl);
                    break;
#endif
                case ConstraintSolverType::NONE:
                default:
                    PINOCCHIO_THROW(std::runtime_error, "Invalid ConstraintSolverType.");
                    break;
                }

                // clang-format off
        #undef SIMPLEX_STEP_LOOP
                // clang-format on
            }

            static void step_wrapper2(
                Self & self,
                const VectorXs & q,
                const VectorXs & v,
                const VectorXs & tau,
                const SpatialForceVector & fext,
                Scalar dt,
                const ConstraintSolverType solver_type,
                std::size_t nsteps)
            {
                // clang-format off
        #define SIMPLEX_STEP_LOOP(solver_pinocchio_type)                                                 \
          self.template step<solver_pinocchio_type>(q, v, tau, fext, dt);                               \
          for (std::size_t i = 0; i < nsteps - 1; ++i)                                                  \
          {                                                                                             \
            self.template step<solver_pinocchio_type>(self.state.qnew, self.state.vnew, tau, fext, dt); \
          }
                // clang-format on

                switch (solver_type)
                {
                case ConstraintSolverType::ADMM:
                    SIMPLEX_STEP_LOOP(::simplex::ADMMContactSolverTpl);
                    break;
                case ConstraintSolverType::PGS:
                    SIMPLEX_STEP_LOOP(::pinocchio::PGSContactSolverTpl);
                    break;
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                case ConstraintSolverType::CLARABEL:
                    SIMPLEX_STEP_LOOP(::simplex::ClarabelContactSolverTpl);
                    break;
#endif
                case ConstraintSolverType::NONE:
                default:
                    PINOCCHIO_THROW(std::runtime_error, "Invalid ConstraintSolverType.");
                    break;
                }

                // clang-format off
        #undef SIMPLEX_STEP_LOOP
                // clang-format on
            }

            static void rollout_wrapper(
                Self & self,
                const VectorXs & q,
                const VectorXs & v,
                const std::vector<VectorXs> & taus,
                Scalar dt,
                const ConstraintSolverType solver_type)
            {
                // clang-format off
        #define SIMPLEX_STEP_LOOP(solver_pinocchio_type)                   \
          {                                                               \
            VectorXs q_ = q;                                              \
            VectorXs v_ = v;                                              \
            for (const auto& tau: taus)                                   \
            {                                                             \
              self.template step<solver_pinocchio_type>(q_, v_, tau, dt); \
              q_ = self.state.qnew;                                       \
              v_ = self.state.vnew;                                       \
            }                                                             \
          }
                // clang-format on

                Py_BEGIN_ALLOW_THREADS;
                switch (solver_type)
                {
                case ConstraintSolverType::ADMM:
                    SIMPLEX_STEP_LOOP(::simplex::ADMMContactSolverTpl);
                    break;
                case ConstraintSolverType::PGS:
                    SIMPLEX_STEP_LOOP(::pinocchio::PGSContactSolverTpl);
                    break;
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                case ConstraintSolverType::CLARABEL:
                    SIMPLEX_STEP_LOOP(::simplex::ClarabelContactSolverTpl);
                    break;
#endif
                case ConstraintSolverType::NONE:
                default:
                    PINOCCHIO_THROW(std::runtime_error, "Invalid ConstraintSolverType.");
                    break;
                }
                Py_END_ALLOW_THREADS;

                // clang-format off
        #undef SIMPLEX_STEP_LOOP
                // clang-format on
            }

            static void rollout_wrapper2(
                Self & self, //
                const VectorXs & q,
                const VectorXs & v,
                const MatrixXs & taus,
                Scalar dt,
                const ConstraintSolverType solver_type)
            {
                // clang-format off
        #define SIMPLEX_STEP_LOOP(solver_pinocchio_type)                           \
          {                                                                       \
            VectorXs q_ = q;                                                      \
            VectorXs v_ = v;                                                      \
            for (int i = 0; i < taus.rows(); ++i)                                 \
            {                                                                     \
              self.template step<solver_pinocchio_type>(q_, v_, taus.row(i), dt); \
              q_ = self.state.qnew;                                               \
              v_ = self.state.vnew;                                               \
            }                                                                     \
          }
                // clang-format on

                Py_BEGIN_ALLOW_THREADS;
                switch (solver_type)
                {
                case ConstraintSolverType::ADMM:
                    SIMPLEX_STEP_LOOP(::simplex::ADMMContactSolverTpl);
                    break;
                case ConstraintSolverType::PGS:
                    SIMPLEX_STEP_LOOP(::pinocchio::PGSContactSolverTpl);
                    break;
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                case ConstraintSolverType::CLARABEL:
                    SIMPLEX_STEP_LOOP(::simplex::ClarabelContactSolverTpl);
                    break;
#endif
                case ConstraintSolverType::NONE:
                default:
                    PINOCCHIO_THROW(std::runtime_error, "Invalid ConstraintSolverType.");
                    break;
                }
                Py_END_ALLOW_THREADS;

                // clang-format off
        #undef SIMPLEX_STEP_LOOP
                // clang-format on
            }

            template<class PyClass>
            void visit(PyClass & cl) const
            {
                cl
                    // -----------------------------
                    // CONSTRUCTORS
                    .def(
                        bp::init<ModelHandle, DataHandle, GeometryModelHandle, GeometryDataHandle>(
                            bp::args("self", "model", "data", "geom_model", "geom_data"), "Constructor"))
                    .def(bp::init<ModelHandle, GeometryModelHandle>(bp::args("self", "model", "geom_model"), "Constructor"))

                    // -----------------------------
                    // ATTRIBUTES
                    .add_property(
                        "model",
                        bp::make_function(
                            +[](Self & self) -> const Model & { return self.model(); }, bp::return_internal_reference<>()),
                        "Pinocchio model.")
                    .add_property(
                        "data",
                        bp::make_function(
                            +[](Self & self) -> const Data & { return self.data(); }, bp::return_internal_reference<>()),
                        "Pinocchio data.")
                    .add_property(
                        "geom_model",
                        bp::make_function(
                            +[](Self & self) -> const GeometryModel & { return self.geom_model(); }, bp::return_internal_reference<>()),
                        "Pinocchio geometry model.")
                    .add_property(
                        "geom_data",
                        bp::make_function(
                            +[](Self & self) -> const GeometryData & { return self.geom_data(); }, bp::return_internal_reference<>()),
                        "Pinocchio geometry data.")

                    .def_readonly("config", &Self::config, "SimulatorX configuration options.")
                    .def_readonly("state", &Self::state, "SimulatorX state.")
                    .def_readonly("workspace", &Self::workspace, "SimulatorX workspace.")
                    .def_readonly("timings", &Self::timings, "SimulatorX timings.")

                    // -----------------------------
                    // METHODS
                    .def(
                        "init", &Self::init, bp::args("self"),
                        "Allocates data for the simulator-x based on model/geom_model. This method should be called if model or geom_model "
                        "have been "
                        "modified (i.e. when the simulated scene is modified).")
                    //
                    .def(
                        "reset", &Self::reset, bp::args("self"),
                        "Resets the simulator to accept a new initial state, i.e. before looping on the `step` method.")
                    //
                    .def(
                        "addPointAnchorConstraints", &Self::addPointAnchorConstraints, bp::args("self"),
                        "Add a vector of point anchor constraints.")
                    //
                    .def(
                        "addFrameAnchorConstraints", &Self::addFrameAnchorConstraints, bp::args("self"),
                        "Add a vector of frame anchor constraints.")
                    //
                    .def("isReset", &Self::isReset, bp::args("self"), "Returns true if the simulator is in its reset state.")
                    //
                    .def("check", &Self::check, bp::args("self"), "Check consistency of the simulator.")
                    //
                    .def(
                        "step", step_wrapper,
                        (bp::arg("self"), bp::arg("q"), bp::arg("v"), bp::arg("tau"), bp::arg("dt"),
                         bp::arg("constraint_solver_type") = ConstraintSolverType::ADMM, bp::arg("nsteps") = 1),
                        "Step the simulator.")
                    .def(
                        "step", step_wrapper2,
                        (bp::arg("self"), bp::arg("q"), bp::arg("v"), bp::arg("tau"), bp::arg("fext"), bp::arg("dt"),
                         bp::arg("constraint_solver_type") = ConstraintSolverType::ADMM, bp::arg("nsteps") = 1),
                        "Step the simulator.")
                    //
                    //
                    .def(
                        "rollout", rollout_wrapper,
                        (bp::arg("self"), bp::arg("q"), bp::arg("v"), bp::arg("taus"), bp::arg("dt"),
                         bp::arg("constraint_solver_type") = ConstraintSolverType::ADMM),
                        "Compute a trajectory by performing a rollout of a policy with a list of taus. This function releases Python GIL "
                        "so it can be "
                        "parallelized. TODO: in the future, rollout should take a control policy.")
                    .def(
                        "rollout", rollout_wrapper2,
                        (bp::arg("self"), bp::arg("q"), bp::arg("v"), bp::arg("taus"), bp::arg("dt"),
                         bp::arg("constraint_solver_type") = ConstraintSolverType::ADMM),
                        "Compute a trajectory by performing a rollout of a policy with a list of taus. This function releases Python GIL "
                        "so it can be "
                        "parallelized. TODO: in the future, rollout should take a control policy.");

                // Register handles
                {
                    // Check registration
                    {
                        const bp::type_info info = bp::type_id<ModelHandle>();
                        const bp::converter::registration * reg = bp::converter::registry::query(info);
                        if (!reg)
                        {
                            bp::register_ptr_to_python<ModelHandle>();
                        }
                    }
                    {
                        const bp::type_info info = bp::type_id<DataHandle>();
                        const bp::converter::registration * reg = bp::converter::registry::query(info);
                        if (!reg)
                        {
                            bp::register_ptr_to_python<DataHandle>();
                        }
                    }
                    {
                        const bp::type_info info = bp::type_id<GeometryModelHandle>();
                        const bp::converter::registration * reg = bp::converter::registry::query(info);
                        if (!reg)
                        {
                            bp::register_ptr_to_python<GeometryModelHandle>();
                        }
                    }
                    {
                        const bp::type_info info = bp::type_id<GeometryDataHandle>();
                        const bp::converter::registration * reg = bp::converter::registry::query(info);
                        if (!reg)
                        {
                            bp::register_ptr_to_python<GeometryDataHandle>();
                        }
                    }
                }
            }

            static void expose_constraint_solvers_configs()
            {

                using ConstraintSolverConfigBase = ConstraintSolverConfigBaseTpl<Scalar>;
                using ADMMConstraintSolverConfig = ADMMConstraintSolverConfigTpl<Scalar>;
                using PGSConstraintSolverConfig = PGSConstraintSolverConfigTpl<Scalar>;
                using ClarabelConstraintSolverConfig = ClarabelConstraintSolverConfigTpl<Scalar>;

                {
                    bp::class_<ConstraintSolverConfigBase>(
                        "ConstraintSolverConfigBase", "Base struct to set up constraint solvers.", bp::no_init)
                        .def_readwrite(
                            "max_iter", &ConstraintSolverConfigBase::max_iter, "Max number of iteration of the constraint solver.")
                        .def_readwrite(
                            "absolute_precision", &ConstraintSolverConfigBase::absolute_precision,
                            "Absolute convergence precision of the constraint solver.")
                        .def_readwrite(
                            "relative_precision", &ConstraintSolverConfigBase::relative_precision,
                            "Relative convergence precision of the constraint solver.")
                        .def_readwrite("stat_record", &ConstraintSolverConfigBase::stat_record, "Record metrics of the constraint solver.");
                }

                {
                    bp::class_<ADMMConstraintSolverConfig, bp::bases<ConstraintSolverConfigBase>>(
                        "ADMMConstraintSolverConfig", "Configuration for the ADMM constraint solver inside `Simulator`.", bp::no_init)
                        .def_readwrite("mu_prox", &ADMMConstraintSolverConfig::mu_prox, "Proximal parameter of ADMM.")
                        .def_readwrite("tau_prox", &ADMMConstraintSolverConfig::tau_prox, "Tau prox.")
                        .def_readwrite(
                            "warmstart_mu_prox", &ADMMConstraintSolverConfig::warmstart_mu_prox,
                            "Whether or not to warm start the proximal value term, using the one from last step.")
                        .def_readwrite(
                            "mu_prox_prev", &ADMMConstraintSolverConfig::mu_prox_prev,
                            "Final value of mu_prox once `solve` has been called. When step is called, this is the value of the previous "
                            "mu_prox.")

                        .def_readwrite("rho", &ADMMConstraintSolverConfig::rho, "Initial value of rho for linear and constant update rule.")
                        .def_readwrite(
                            "tau", &ADMMConstraintSolverConfig::tau,
                            "ADMM augmented lagragian penalty is tau * rho (rho is scaled during iterations).")
                        .def_readwrite(
                            "warmstart_rho", &ADMMConstraintSolverConfig::warmstart_rho,
                            "Whether or not rho should be warmstarted (using rho of previous timestep). If true, then rho will be changed "
                            "by the solver "
                            "once `solve` has been called.")
                        .def_readwrite(
                            "rho_prev", &ADMMConstraintSolverConfig::rho_prev,
                            "Final value of rho once `solve` has been called. When step is called, this is the value of the previous rho.")

                        .def_readwrite(
                            "lanczos_size", &ADMMConstraintSolverConfig::lanczos_size,
                            "Size of Lanczos decomposition. Higher yields more accurate delassus eigenvalues estimates.")
                        .def_readwrite("rho_momentum", &ADMMConstraintSolverConfig::rho_momentum, "Momentum of rho (0 is no momentum).")
                        .def_readwrite(
                            "ratio_primal_dual", &ADMMConstraintSolverConfig::ratio_primal_dual,
                            "Ratio above/below which to trigger the rho update.")
                        .def_readwrite(
                            "max_delassus_decomposition_updates", &ADMMConstraintSolverConfig::max_delassus_decomposition_updates,
                            "Maximum number of delassus decomposition updates.")
                        .def_readwrite(
                            "dual_momentum", &ADMMConstraintSolverConfig::dual_momentum,
                            "Momentum value on the dual variable (0 is no momentum).")
                        .def_readwrite(
                            "rho_update_ratio", &ADMMConstraintSolverConfig::rho_update_ratio,
                            "The rho is only updated if the ratio between the current rho and the new one is bigger/lower than a threshold "
                            "ratio.")
                        .def_readwrite(
                            "admm_update_rule", &ADMMConstraintSolverConfig::admm_update_rule,
                            "Update rule for the ADMM constraint solver (constant, linear, spectral or osqp).")
                        .def_readwrite(
                            "rho_min_update_frequency", &ADMMConstraintSolverConfig::rho_min_update_frequency,
                            "The solver must wait this amount of iters to trigger a new rho update. Must be >= 1.")
                        .def_readwrite(
                            "anderson_acceleration_capacity", &ADMMConstraintSolverConfig::anderson_acceleration_capacity,
                            "Amount of history needed to trigger the anderson acceleration. A history >= 2 will trigger the acceleration.")

                        .def_readwrite(
                            "rho_power", &ADMMConstraintSolverConfig::rho_power,
                            "Initial value of rho_power when using the spectral update rule.")
                        .def_readwrite("rho_power_factor", &ADMMConstraintSolverConfig::rho_power_factor, "Update factor on rho_power.")

                        .def_readwrite(
                            "linear_update_rule_factor", &ADMMConstraintSolverConfig::linear_update_rule_factor,
                            "Update factor on rho when using linear update rule.");
                }

                {
                    bp::class_<PGSConstraintSolverConfig, bp::bases<ConstraintSolverConfigBase>>(
                        "PGSConstraintSolverConfig", "Configuration for the PGS constraint solver inside `Simulator`.", bp::no_init)
                        .def_readwrite(
                            "over_relax", &PGSConstraintSolverConfig::over_relax, "Optional over relaxation value, default to 1.");
                }

                {
                    bp::class_<ClarabelConstraintSolverConfig, bp::bases<ConstraintSolverConfigBase>>(
                        "ClarabelConstraintSolverConfig", "Configuration for the Clarabel constraint solver inside `SimulatorX`.",
                        bp::no_init)
                        .def_readwrite("tol_feas", &ClarabelConstraintSolverConfig::tol_feas, "Tolerance for feasibility checks.")
                        .def_readwrite("tol_ktratio", &ClarabelConstraintSolverConfig::tol_ktratio, "Tolerance for KKT conditions.")
                        .def_readwrite("verbose", &ClarabelConstraintSolverConfig::verbose, "Enable verbose output.");
                }
            }

            static void expose()
            {
                {
                    bp::class_<ConstraintSolverConfigContainer>(
                        "ConstraintSolverConfigContainer", "Configuration for the simulator's constraint solvers.", bp::no_init)
                        .def_readwrite(
                            "admm_config", &ConstraintSolverConfigContainer::admm_config, "Configuration for the ADMM constraint solver.")
                        .def_readwrite(
                            "pgs_config", &ConstraintSolverConfigContainer::pgs_config, "Configuration for the PGS constraint solver.")
                        .def_readwrite(
                            "clarabel_config", &ConstraintSolverConfigContainer::clarabel_config,
                            "Configuration for the Clarabel constraint solver.");
                }

                {
                    bp::class_<SimulatorConfig>("SimulatorConfig", "Options to configurate the simulator.", bp::no_init)
                        .def_readwrite(
                            "constraint_solvers_configs", &SimulatorConfig::constraint_solvers_configs,
                            "Configuration of the constraint solvers that can be used in `step`.")
                        .def_readwrite(
                            "warmstart_constraint_velocities", &SimulatorConfig::warmstart_constraint_velocities,
                            "Whether or not to warm-start the dual variable (constraint velocities) of the constraint solver.")
                        .def_readwrite(
                            "measure_timings", &SimulatorConfig::measure_timings,
                            "Whether or not timings of the `step` function are measured. If set to true, the timing of the last call to "
                            "`step` can be "
                            "accessed in `SimulatorX.timings`.");
                }

                expose_constraint_solvers_configs();

                {
                    bp::enum_<ConstraintSolverType>("ConstraintSolverType")
                        .value("PGS", ConstraintSolverType::PGS)
                        .value("ADMM", ConstraintSolverType::ADMM)
                        .value("CLARABEL", ConstraintSolverType::CLARABEL)
                        .value("NONE", ConstraintSolverType::NONE);
                }

                {
                    bp::class_<SimulatorState>(
                        "SimulatorState",
                        "State of the simulator. When `step` is called with (q, v, tau, fext, dt), these quantities are stored in "
                        "the state of the simulator. During `step`, the rest of the state is computed by solving the equations of "
                        "motion s.t. the dynamics' constraints are fulfilled.")
                        .def_readonly("q", &SimulatorState::q, "Joints configuration of the system - copied from `step` input.")
                        .def_readonly("v", &SimulatorState::v, "Joints velocity of the system - copied from `step` input.")
                        .def_readonly("tau", &SimulatorState::tau, "Joint torques applied to the system - copied from `step` input.")
                        .def_readonly("dt", &SimulatorState::dt, "Time step of the simulator - copied from `step` input.")
                        .def_readonly(
                            "fext", &SimulatorState::fext,
                            "External 6D forces exerted in the local frame of each joint - copied from `step` input.")
                        .def_readonly(
                            "constraint_solver_type", &SimulatorState::constraint_solver_type,
                            "Type of constraint solver used in the current call to the `step` method. This enum is meant to indicate which "
                            "solver is "
                            "currently used. It does not reflect **which one to use**. To use a specific constraint solver, use the "
                            "dedicated `step` "
                            "method (e.g. step (default ADMM constraint solver) or stepPGS (uses the PGS constraint solver)).")
                        .def_readonly("tau_damping", &SimulatorState::tau_damping, "Torque due to damping - computed as -damping * v.")
                        .def_readonly("qnew", &SimulatorState::qnew, "The updated joint configuration of the system.")
                        .def_readonly(
                            "vfree", &SimulatorState::vfree,
                            "Free velocity i.e. the updated velocity of the system without any constraint forces.")
                        .def_readonly(
                            "afree", &SimulatorState::afree,
                            "Free acceleration i.e. the updated velocity of the system without any constraint forces.")
                        .def_readonly(
                            "vnew", &SimulatorState::vnew,
                            "The updated velocity, taking into account the correction due to constraint forces.")
                        .def_readonly(
                            "anew", &SimulatorState::anew,
                            "The updated acceleration, taking into account the correction due to constraint forces.")
                        .def_readonly("tau_total", &SimulatorState::tau_total, "Total torques applied on joints.")
                        .def_readonly("tau_constraints", &SimulatorState::tau_constraints, "Torques applied on joints due to constraints.")
                        .def_readonly(
                            "is_reset", &SimulatorState::is_reset, "Whether or not the simulator is in reset state (no previous state).")

                        .def("init", &SimulatorState::init, "Initialize simulator's state.")
                        .def(
                            "reset", &SimulatorState::reset,
                            "Reset the simulator's state, preparing it for a new trajectory within the same scene (same "
                            "model/geom_model/constraints).")
                        .def("check", &SimulatorState::check, "Sanity check of the simulator's state.");
                }

                {
                    bp::class_<ConstraintSolverContainer>(
                        "ConstraintSolverContainer",
                        "Container for the constraint solvers that can be used to solve constraints during `step`.", bp::no_init)
                        .def_readonly("admm_solver", &ConstraintSolverContainer::admm_solver, "ADMM constraint solver.")
                        .def_readonly("pgs_solver", &ConstraintSolverContainer::pgs_solver, "PGS constraint solver.")
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                        .def_readonly(
                            "clarabel_solver", &ConstraintSolverContainer::clarabel_solver,
                            "Constraint solver based on Clarabel IPM solver")
#endif
                        ;
                }

                {
                    bp::class_<SimulatorWorkspace>(
                        "SimulatorWorkspace",
                        "Workspace of the simulator. Holds tools required to do constrained physics computation (collision "
                        "detection and constraint resolution).")
                        .add_property(
                            "collision_callback",
                            bp::make_function(
                                +[](SimulatorWorkspace & workspace) -> const CollisionCallBackCollect & {
                                    return workspace.collision_callback();
                                },
                                bp::return_internal_reference<>()),
                            "Broad phase collision callback.")
                        .add_property(
                            "broadphase_manager",
                            bp::make_function(
                                +[](SimulatorWorkspace & workspace) -> const BroadPhaseManager & { return workspace.broadphase_manager(); },
                                bp::return_internal_reference<>()),
                            "Broad phase manager.")
                        .add_property(
                            "constraint_problem",
                            bp::make_function(
                                +[](SimulatorWorkspace & workspace) -> const ConstraintsProblemDerivatives & {
                                    return workspace.constraint_problem();
                                },
                                bp::return_internal_reference<>()),
                            "Constraint problem.")
                        .def_readonly(
                            "constraint_solvers", &SimulatorWorkspace::constraint_solvers,
                            "Container for the constraint solvers that can be used to solve constraints during `step`.")

                        .def("init", &SimulatorWorkspace::init, "Initialize simulator workspace.")
                        .def(
                            "reset", &SimulatorWorkspace::reset,
                            "Reset the simulator's workspace, preparing it for a new trajectory within the same scene (same "
                            "model/geom_model/constraints).")
                        .def("check", &SimulatorWorkspace::check, "Sanity check of the simulator's workspace.");
                }

                {
                    bp::class_<SimulatorTimings>("SimulatorTimings", "Timings related to events occuring during a call to `step`.")
                        .def_readonly(
                            "timings_broadphase_collision_detection", &SimulatorTimings::timings_broadphase_collision_detection,
                            "Timings for broad phase collision detection.")
                        .def_readonly(
                            "timings_narrowphase_collision_detection", &SimulatorTimings::timings_narrowphase_collision_detection,
                            "Timings for narrow phase collision detection.")
                        .def_readonly(
                            "timings_collision_detection", &SimulatorTimings::timings_collision_detection,
                            "Timings for collision detection.")
                        .def_readonly(
                            "timings_constraint_solver", &SimulatorTimings::timings_constraint_solver,
                            "Timings for the call to the constraint solver.")
                        .def_readonly("timings_step", &SimulatorTimings::timings_step, "Timings for the whole `step` method.");
                }

                {
                    bp::class_<SimulatorX>("SimulatorX", "Instance of SimulatorX.", bp::no_init)
                        .def(SimulatorXPythonVisitor<SimulatorX>())
                        .def(::pinocchio::python::CopyableVisitor<SimulatorX>());
                }
            }
        };

    } // namespace python
} // namespace simplex

#endif // ifndef __simplex_python_core_simulator_x_hpp__
