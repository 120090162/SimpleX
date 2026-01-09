#ifndef __simplex_core_simulator_x_hpp__
#define __simplex_core_simulator_x_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/core/constraints-problem-derivatives.hpp"
#include "simplex/solver/admm-solver.hpp"
#include "simplex/solver/clarabel-solver.hpp"
#include "simplex/macros.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/contact-inverse-dynamics.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/pgs-solver.hpp>
#include <pinocchio/algorithm/proximal.hpp>
#include "pinocchio/algorithm/delassus-operator-cholesky-expression.hpp"
#include <pinocchio/algorithm/delassus-operator-dense.hpp>
#include <pinocchio/algorithm/delassus-operator-rigid-body.hpp>
#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/constraints/utils.hpp>

#include <pinocchio/collision/collision.hpp>
#include <pinocchio/collision/broadphase.hpp>
#include <pinocchio/collision/broadphase-manager.hpp>

#include <coal/broadphase/broadphase_dynamic_AABB_tree.h>

namespace simplex
{
    /// Constraint solver config forward declarations.
    template<typename _Scalar>
    struct ADMMConstraintSolverConfigTpl;

    template<typename _Scalar>
    struct PGSConstraintSolverConfigTpl;

    template<typename _Scalar>
    struct ClarabelConstraintSolverConfigTpl;
    /**
     * @brief Traits to define associated types for the SimulatorX.
     */
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct traits<SimulatorXTpl<_Scalar, _Options, JointCollectionTpl>>
    {
        using Scalar = _Scalar;
    };

    /**
     * @brief Main simulator class managing the physics loop.
     *
     * It handles the interaction between Pinocchio's rigid body dynamics,
     * FCL's collision detection, and various contact solvers.
     */
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct SimulatorXTpl
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // -------------------------------------------------------------------------------------------------
        // TYPEDEFS
        // -------------------------------------------------------------------------------------------------
        using Scalar = _Scalar;
        enum
        {
            Options = _Options
        };

        using GeomIndex = pinocchio::GeomIndex;

        using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
        using Vector3s = Eigen::Matrix<Scalar, 3, 1, Options>;
        using Vector6s = Eigen::Matrix<Scalar, 6, 1, Options>;
        using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;

        using Model = ::pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl>;
        using ModelHandle = std::shared_ptr<Model>;
        using Data = typename Model::Data;
        using DataHandle = std::shared_ptr<Data>;
        using GeometryModel = ::pinocchio::GeometryModel;
        using GeometryModelHandle = std::shared_ptr<GeometryModel>;
        using GeometryData = ::pinocchio::GeometryData;
        using GeometryDataHandle = std::shared_ptr<GeometryData>;

        using Force = typename Data::Force;
        using Motion = typename Data::Motion;
        using SE3 = typename Data::SE3;
        using JointModel = ::pinocchio::JointModelTpl<Scalar, Options, JointCollectionTpl>;

        // Broadphase for collision detection optimization
        // TODO: template simulator by broad phase manager
        using BroadPhaseManager = ::pinocchio::BroadPhaseManagerTpl<coal::DynamicAABBTreeCollisionManager>;
        using BroadPhaseManagerHandle = std::shared_ptr<BroadPhaseManager>;
        using CollisionCallBackCollect = ::pinocchio::CollisionCallBackCollect;
        using CollisionCallBackCollectHandle = std::shared_ptr<CollisionCallBackCollect>;

        // Contact solvers
        using ADMMConstraintSolver = ::simplex::ADMMContactSolverTpl<Scalar>;
        using ADMMConstraintSolverConfig = ADMMConstraintSolverConfigTpl<Scalar>;
        using PGSConstraintSolver = ::pinocchio::PGSContactSolverTpl<Scalar>;
        using PGSConstraintSolverConfig = PGSConstraintSolverConfigTpl<Scalar>;
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
        using ClarabelConstraintSolver = ::simplex::ClarabelContactSolverTpl<Scalar, Options>;
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT
        using ClarabelConstraintSolverConfig = ClarabelConstraintSolverConfigTpl<Scalar>;

        using SpatialForce = ::pinocchio::ForceTpl<Scalar, Options>;
        using SpatialForceVector = PINOCCHIO_ALIGNED_STD_VECTOR(SpatialForce);

        using ConstraintsProblemDerivatives = ConstraintsProblemDerivativesTpl<Scalar, Options, JointCollectionTpl>;
        using ConstraintsProblemDerivativesHandle = std::shared_ptr<ConstraintsProblemDerivatives>;

        // Specific constraint models
        using FrictionalPointConstraintModel = typename ConstraintsProblemDerivatives::FrictionalPointConstraintModel;
        using FrictionalJointConstraintModel = typename ConstraintsProblemDerivatives::FrictionalJointConstraintModel;
        using JointLimitConstraintModel = typename ConstraintsProblemDerivatives::JointLimitConstraintModel;
        using BilateralPointConstraintModel = typename ConstraintsProblemDerivatives::BilateralPointConstraintModel;
        using BilateralPointConstraintData = typename ConstraintsProblemDerivatives::BilateralPointConstraintData;
        using BilateralPointConstraintModelVector = typename ConstraintsProblemDerivatives::BilateralPointConstraintModelVector;
        using WeldConstraintModel = typename ConstraintsProblemDerivatives::WeldConstraintModel;
        using WeldConstraintData = typename ConstraintsProblemDerivatives::WeldConstraintData;
        using WeldConstraintModelVector = typename ConstraintsProblemDerivatives::WeldConstraintModelVector;

        using ConstraintModel = ::pinocchio::ConstraintModelTpl<Scalar, Options>;
        using ConstraintModelVector = PINOCCHIO_ALIGNED_STD_VECTOR(ConstraintModel);
        using WrappedConstraintModel = std::reference_wrapper<const ConstraintModel>;
        using WrappedConstraintModelVector = PINOCCHIO_ALIGNED_STD_VECTOR(WrappedConstraintModel);

        using ConstraintData = ::pinocchio::ConstraintDataTpl<Scalar, Options>;
        using ConstraintDataVector = PINOCCHIO_ALIGNED_STD_VECTOR(ConstraintData);
        using WrappedConstraintData = std::reference_wrapper<ConstraintData>;
        using WrappedConstraintDataVector = PINOCCHIO_ALIGNED_STD_VECTOR(WrappedConstraintData);

        using ConstraintCholeskyDecomposition = ::pinocchio::ContactCholeskyDecompositionTpl<Scalar, Options>;
        using DelassusCholeskyExpressionOperator = ::pinocchio::DelassusCholeskyExpressionTpl<ConstraintCholeskyDecomposition>;
        using DelassusRigidBodyOperator =
            ::pinocchio::DelassusOperatorRigidBodySystemsTpl<Scalar, Options, JointCollectionTpl, WrappedConstraintModel>;
        using DelassusDenseOperator = ::pinocchio::DelassusOperatorDenseTpl<Scalar>;
        using DelassusType = typename ConstraintsProblemDerivatives::DelassusType;

        // -------------------------------------------------------------------------------------------------
        // MEMBERS
        // -------------------------------------------------------------------------------------------------
        SIMPLEX_PROTECTED
        /// \brief Handle to the model of the system.
        /// \note Can be accessed by reference via `model()` method.
        ModelHandle model_;

        /// \brief Handle to the model's data of the system.
        /// \note Can be accessed by reference via `model()` method.
        DataHandle data_;

        /// \brief Handle to the geometry model of the system.
        /// \note Can be accessed by reference via `model()` method.
        GeometryModelHandle geom_model_;

        /// \brief Handle to the geometry model's data of the system.
        /// \note Can be accessed by reference via `model()` method.
        GeometryDataHandle geom_data_;

        SIMPLEX_PUBLIC
        ///
        /// \brief Configuration of the simulator.
        /// \note Modify the configuration of the simulator to change its behavior.
        struct SimulatorConfig
        {
            /// \brief Configuration of the simulator's constraint solvers.
            struct ConstraintSolverConfigContainer
            {
                /// \brief Configuration for the ADMM constraint solver.
                ADMMConstraintSolverConfig admm_config;

                /// \brief Configuration for the PGS constraint solver.
                PGSConstraintSolverConfig pgs_config;

                /// \brief Configuration for the Clarabel constraint solver.
                ClarabelConstraintSolverConfig clarabel_config;

            } constraint_solvers_configs;

            /// \brief Whether or not to warm-start the dual variable (constraint velocities) of the constraint solver.
            bool warmstart_constraint_velocities{true};

            /// \brief Whether or not timings of the `step` function are measured.
            /// If set to true, the timing of the last call to `step` can be accessed in `SimulatorXTpl::timings`.
            bool measure_timings{false};
        } config;

        ///
        /// \brief State of the simulator.
        /// When `step` is called with (q, v, tau, fext, dt), these quantities are stored in the state of the simulator.
        /// During `step`, the rest of the state is computed by solving the equations of motion s.t. the dynamics' constraints are
        /// fulfilled.
        struct SimulatorState
        {
            /// \brief Joints configuration of the system - copied from `step` input.
            VectorXs q;

            /// \brief Joints velocity of the system - copied from `step` input.
            VectorXs v;

            /// \brief Joint torques applied to the system - copied from `step` input.
            VectorXs tau;

            /// \brief Time step of the simulator - copied from `step` input.
            Scalar dt{Scalar(-1)};

            /// \brief External 6D forces exerted in the local frame of each joint - copied from `step` input.
            SpatialForceVector fext;

            /// \brief Type of constraint solver used in the current call to the `step` method.
            /// This enum is meant to indicate which solver is currently used. It does not reflect **which one to use**.
            /// To use a specific constraint solver, use the template of the `step` method (e.g. step<PGSConstraintSolver>,
            /// step<ADMMConstraintSolver>, or step<ClarabelConstraintSolver>).
            enum struct ConstraintSolverType
            {
                PGS,
                ADMM,
                CLARABEL,
                NONE
            } constraint_solver_type{ConstraintSolverType::NONE};

            /// \brief Torque due to damping - computed as -damping * v.
            VectorXs tau_damping;

            /// \brief The updated joint configuration of the system.
            VectorXs qnew;

            /// \brief Free velocity i.e. the updated velocity of the system without any constraint forces.
            VectorXs vfree;

            /// \brief Free acceleration i.e. the updated acceleration of the system without any constraint forces.
            VectorXs afree;

            /// \brief The updated velocity, taking into account the correction due to constraint forces.
            VectorXs vnew;

            /// \brief The updated acceleration, taking into account the correction due to constraint forces.
            VectorXs anew;

            /// \brief Total torques (i.e. applied tau + dry friction on joints + forces due to joint limits)
            /// applied on joints.
            VectorXs tau_total;

            /// \brief Torques applied on joints due to constraints.
            VectorXs tau_constraints;

            /// \brief Whether or not the simulator is in reset state (no previous state).
            bool is_reset{true};

            ///
            /// \brief Initialize simulator's state.
            void init(int nq, int nv, int njoints);

            ///
            /// \brief Reset the simulator's state, preparing it for a new trajectory within the same scene (same
            /// model/geom_model/constraints).
            void reset();

            ///
            /// \brief Sanity check of the simulator's state.
            bool check(const Model & model) const;
        } state;

        /// \brief Workspace of the simulator.
        /// Holds tools required to do constrained physics computation (collision detection and constraint resolution).
        struct SimulatorWorkspace
        {
            SIMPLEX_PROTECTED
            /// \brief Collision callback for broadphase collision detection.
            CollisionCallBackCollectHandle collision_callback_;

            /// \brief Broad phase manager.
            /// \note Can be accessed by reference via `broadphase_manager()` method.
            BroadPhaseManagerHandle broadphase_manager_;

            /// \brief Constraint problem.
            /// \note Can be accessed by reference via `constraint_problem()` method.
            ConstraintsProblemDerivativesHandle constraint_problem_;

            SIMPLEX_PUBLIC
            ///
            /// \brief Container for the constraint solvers and their associated results that can be used to solve constraints during
            /// `step`.
            struct ConstraintSolverContainer
            {
                /// \brief ADMM constraint solver.
                ADMMConstraintSolver admm_solver;

                /// \brief PGS constraint solver.
                PGSConstraintSolver pgs_solver;

#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                /// \brief Clarabel constraint solver.
                ClarabelConstraintSolver clarabel_solver;
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

                /// \brief Base constructor creates empty ADMM, PGS, and Clarabel solvers.
                /// This allows the simulator to create the solvers only when needed.
                ConstraintSolverContainer()
                : admm_solver(0)
                , pgs_solver(0)
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
                , clarabel_solver(0)
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT
                {
                }
            } constraint_solvers;

            /// \brief Temporary variable used to integrate the system's state.
            VectorXs vnew_integration_tmp;

            ///
            /// \brief Returns a const reference to the broad phase collision callback.
            const CollisionCallBackCollect & collision_callback() const
            {
                assert(collision_callback_ != nullptr && "collision_callback_ is nullptr");
                return ::pinocchio::helper::get_ref(collision_callback_);
            }

            ///
            /// \brief Returns a const reference to the broad phase collision callback.
            CollisionCallBackCollect & collision_callback()
            {
                assert(collision_callback_ != nullptr && "collision_callback_ is nullptr");
                return ::pinocchio::helper::get_ref(collision_callback_);
            }

            ///
            /// \brief Returns a const reference to the broad phase manager.
            const BroadPhaseManager & broadphase_manager() const
            {
                assert(broadphase_manager_ != nullptr && "broadphase_manager_ is nullptr");
                return ::pinocchio::helper::get_ref(broadphase_manager_);
            }

            ///
            /// \brief Returns a reference to the broad phase manager.
            BroadPhaseManager & broadphase_manager()
            {
                assert(broadphase_manager_ != nullptr && "broadphase_manager_ is nullptr");
                return ::pinocchio::helper::get_ref(broadphase_manager_);
            }

            ///
            /// \brief Returns a const reference to the constraint problem.
            const ConstraintsProblemDerivatives & constraint_problem() const
            {
                assert(constraint_problem_ != nullptr && "constraint_problem_ is nullptr");
                return ::pinocchio::helper::get_ref(constraint_problem_);
            }

            ///
            /// \brief Returns a const reference to the constraint problem.
            ConstraintsProblemDerivatives & constraint_problem()
            {
                assert(constraint_problem_ != nullptr && "constraint_problem_ is nullptr");
                return ::pinocchio::helper::get_ref(constraint_problem_);
            }

            ///
            /// \brief Returns a handle to the constraint problem
            /// TODO(louis): remove this method
            ConstraintsProblemDerivativesHandle getConstraintProblemHandle() const
            {
                assert(constraint_problem_ != nullptr && "constraint_problem_ is nullptr");
                return constraint_problem_;
            }

            ///
            /// \brief Initialize simulator workspace.
            void init(ModelHandle model, DataHandle data, GeometryModelHandle geom_model, GeometryDataHandle geom_data);

            ///
            /// \brief Reset the simulator's workspace, preparing it for a new trajectory within the same scene (same
            /// model/geom_model/constraints).
            void reset();

            ///
            /// \brief Sanity check of the simulator's workspace.
            bool check(const Model & model, const bool constraint_problem_has_been_updated) const;
        } workspace;

        /// \brief Timings related to events occuring during a call to `step`.
        struct SimulatorTimings
        {
            /// \brief Timings for broad phase collision detection.
            ::coal::CPUTimes timings_broadphase_collision_detection;

            /// \brief Timings for narrow phase collision detection.
            ::coal::CPUTimes timings_narrowphase_collision_detection;

            /// \brief Timings for collision detection.
            ::coal::CPUTimes timings_collision_detection;

            /// \brief Timings for the call to the constraint solver.
            ::coal::CPUTimes timings_constraint_solver;

            /// \brief Timings for the whole `step` method.
            ::coal::CPUTimes timings_step;

            SIMPLEX_PROTECTED
            /// \brief Timer used for the whole `step` method.
            ::coal::Timer timer_step{false};

            /// \brief Timer used for internal methods inside `step`.
            ::coal::Timer timer_internal{false};

            /// \brief Timer used for broad and narrow phase collision detection.
            ::coal::Timer timer_collision_detection{false};

            SIMPLEX_PUBLIC
            /// \brief Clear timings.
            void clear();

            // allow simulator to access timings
            friend struct SimulatorXTpl;
        } timings;

        // -------------------------------------------------------------------------------------------------
        // CONSTRUCTORS
        // -------------------------------------------------------------------------------------------------
        SIMPLEX_PUBLIC
        ///
        /// \brief Constructor specifying the Data and GeometryData associated to the Model and GeometryModel.
        SimulatorXTpl(
            ModelHandle model_handle,              //
            DataHandle data_handle,                //
            GeometryModelHandle geom_model_handle, //
            GeometryDataHandle geom_data_handle);

        ///
        /// \brief Constructor specifying only the Model and GeometryModel.
        SimulatorXTpl(ModelHandle model_handle, GeometryModelHandle geom_model_handle);

        /// \brief Virtual destructor of the base class
        virtual ~SimulatorXTpl()
        {
        }

        // -------------------------------------------------------------------------------------------------
        // HELPERS
        // -------------------------------------------------------------------------------------------------

        /// \brief Allocates data for the simulator based on model/geom_model.
        /// This method should be called if model or geom_model have been modified (i.e. when the simulated scene
        /// is modified).
        void init();

        ///
        /// \brief Resets the internal quantities of the simulator.
        /// This methods can be called before looping on the `step` method, e.g. when the
        /// initial state (q0, v0) of the system is used as an input to `step` and (q0, v0) have not
        /// been computed using the `step` method.
        /// \note If, instead, the simulator's model, data, geom_model or geom_data have changed,
        /// please call a constructor or `init`.
        void reset();

        ///
        /// \brief Helper to add a point anchor constraint.
        void addPointAnchorConstraints(const BilateralPointConstraintModelVector & point_anchor_constraint_models);

        ///
        /// \brief Helper to add a frame anchor constraint.
        void addFrameAnchorConstraints(const WeldConstraintModelVector & frame_anchor_constraint_models);

        /// -------------------------------------------------------------------------------------------------
        /// CORE METHODS
        /// -------------------------------------------------------------------------------------------------

        ///
        /// \brief Main step function of the simulator.
        template<
            template<typename> class ConstraintSolver = ::simplex::ADMMContactSolverTpl,
            typename ConfigVectorType,
            typename VelocityVectorType,
            typename TorqueVectorType>
        void step(
            const Eigen::MatrixBase<ConfigVectorType> & q,
            const Eigen::MatrixBase<VelocityVectorType> & v,
            const Eigen::MatrixBase<TorqueVectorType> & tau,
            Scalar dt);

        ///
        /// \brief Main step function of the simulator.
        /// fext must be expressed in the local frame of each joint.
        // TODO: remove aligned_vector and template by Allocator.
        // TODO: change MatrixBase to PlainObject
        template<
            template<typename> class ConstraintSolver = ::simplex::ADMMContactSolverTpl,
            typename ConfigVectorType,
            typename VelocityVectorType,
            typename TorqueVectorType,
            typename ForceDerived>
        void step(
            const Eigen::MatrixBase<ConfigVectorType> & q,
            const Eigen::MatrixBase<VelocityVectorType> & v,
            const Eigen::MatrixBase<TorqueVectorType> & tau,
            const pinocchio::container::aligned_vector<ForceDerived> & fext,
            const Scalar dt);

        ///
        /// \brief Returns true if the simulator is in its reset state.
        bool isReset() const
        {
            return state.is_reset;
        }

        ///
        /// \brief Check consistency of the simulator.
        bool check(const bool constraint_problem_has_been_updated) const;

        ///
        /// \brief Check consistency of the collision pairs.
        /// This method checks thath the geometry model does not contain collision pairs between
        /// geometry objects from the same parent joint.
        bool checkCollisionPairs() const;

        ///
        /// \brief Returns a const reference to the model
        const Model & model() const
        {
            return ::pinocchio::helper::get_ref(model_);
        }

        ///
        /// \brief Returns a reference to the model
        Model & model()
        {
            return ::pinocchio::helper::get_ref(model_);
        }

        ///
        /// \brief Returns a const reference to the data
        const Data & data() const
        {
            return ::pinocchio::helper::get_ref(data_);
        }

        ///
        /// \brief Returns a reference to the data
        Data & data()
        {
            return ::pinocchio::helper::get_ref(data_);
        }

        ///
        /// \brief Returns a const reference to the geometry model
        const pinocchio::GeometryModel & geom_model() const
        {
            return ::pinocchio::helper::get_ref(geom_model_);
        }

        ///
        /// \brief Returns a reference to the geometry model
        pinocchio::GeometryModel & geom_model()
        {
            return ::pinocchio::helper::get_ref(geom_model_);
        }

        ///
        /// \brief Returns a const reference to the geometry data
        const pinocchio::GeometryData & geom_data() const
        {
            return ::pinocchio::helper::get_ref(geom_data_);
        }

        ///
        /// \brief Returns a reference to the geometry data
        pinocchio::GeometryData & geom_data()
        {
            return ::pinocchio::helper::get_ref(geom_data_);
        }

        SIMPLEX_PROTECTED
        ///
        /// \brief Collision detection
        void detectCollisions();

        ///
        /// \brief Preambule function meant to be run before resolving collisions.
        virtual void preambleResolveConstraints()
        {
        }

        ///
        /// \brief Constraint resolution
        template<template<typename> class ConstraintSolver>
        void resolveConstraints();
    }; // struct SimulatorXTpl

    ///
    /// \brief Base struct to set up constraint solvers.
    template<typename _Scalar>
    struct ConstraintSolverConfigBaseTpl
    {
        using Scalar = _Scalar;

        int max_iter{1000};
        Scalar absolute_precision{1e-8};
        Scalar relative_precision{1e-8};
        bool stat_record{false};
    };

    ///
    /// \brief Configuration of the ADMM constraint solver.
    ///
    /// \note Pinocchio's ADMM has two regularization term
    /// -> tau * rho for the augmented lagrangian term (consensus penalty)
    /// -> mu term for the proximal term (primal variable regularization)
    ///
    /// ADMM update rule: if ratio_primal_dual reached, rho *= rho_increment, where
    /// rho_increment = pow(L/m, rho_power_factor).
    ///
    /// Initialization of rho:
    /// -- If linear update rule   -> rho is initialized by ADMM constructor or `setRho`
    /// -- If spectral update rule -> rho is initialized by ADMM `computeRho` = sqrt(L*m) * pow(L/m, rho_power)
    ///
    template<typename _Scalar>
    struct ADMMConstraintSolverConfigTpl : ConstraintSolverConfigBaseTpl<_Scalar>
    {
        using Scalar = _Scalar;
        using Base = ConstraintSolverConfigBaseTpl<_Scalar>;
        using Base::absolute_precision;
        using Base::max_iter;
        using Base::relative_precision;
        using Base::stat_record;

        // Proximal term
        Scalar mu_prox{1e-6};          // Value of proximal term used if mu_prox is not warm started
        Scalar tau_prox{1};            // Linear scaling factor for mu_prox.
        bool warmstart_mu_prox{false}; // Whether or not mu_prox should be warmstarted (using mu_prox of previous timestep).
                                       // If true, then mu_prox_prev will be changed by the solver once `solve` has been called.
        // REFACTOR this should not be in config
        mutable Scalar mu_prox_prev{1e-6}; // Final value of mu_prox once `solve` has been called.
                                           // When step is called, this is the value of the previous mu_prox.
        // Augmented lagrangian rho term
        Scalar rho{10};           // Initial value of rho for linear and constant update rule.
        Scalar tau{1};            // linear scaling factor for rho.
        bool warmstart_rho{true}; // Whether or not rho should be warmstarted (using rho of previous timestep).
                                  // If true, then rho_prev will be changed by the solver once `solve` has been called.
        // REFACTOR this should not be in config
        mutable Scalar rho_prev{10}; // Final value of rho once `solve` has been called.
                                     // When step is called, this is the value of the previous rho.
        //
        // Common to all admm update rules
        int lanczos_size{10};         // Higher leads to more accurate eigvalues estimation. Max size = constraint problem size
        Scalar rho_momentum{0};       // In [0, 1]. 0 is no momentum.
        Scalar ratio_primal_dual{10}; // A new rho is computed when the primal dual ratio exceeds this threshold
        int max_delassus_decomposition_updates{std::numeric_limits<int>::infinity()};
        Scalar dual_momentum{0};         // In [0, 1], 0 is no momentum.
        Scalar rho_update_ratio{0};      // Must be positive
                                         // The rho update triggers when a new rho is computed and the rhos ratio exceeds this threshold
        int rho_min_update_frequency{1}; // Wait this amount of iters to trigger a new rho update. Must be >= 1
        std::size_t anderson_acceleration_capacity{0}; // Amount of history needed to trigger the anderson acceleration.
                                                       // A history >= 2 will trigger the acceleration.
        ::simplex::ADMMUpdateRule admm_update_rule{::simplex::ADMMUpdateRule::SPECTRAL};
        //
        // Spectral rule settings
        Scalar rho_power{0.2};
        Scalar rho_power_factor{0.05};
        // Linear rule settings
        Scalar linear_update_rule_factor{2};
    };

    ///
    /// \brief Configuration of the PGS constraint solver.
    template<typename _Scalar>
    struct PGSConstraintSolverConfigTpl : ConstraintSolverConfigBaseTpl<_Scalar>
    {
        using Scalar = _Scalar;

        Scalar over_relax{1.0};
    };

    ///
    /// \brief Configuration of the Clarabel constraint solver.
    template<typename _Scalar>
    struct ClarabelConstraintSolverConfigTpl : ConstraintSolverConfigBaseTpl<_Scalar>
    {
        using Scalar = _Scalar;
        using Base = ConstraintSolverConfigBaseTpl<_Scalar>;
        using Base::absolute_precision;
        using Base::max_iter;
        using Base::relative_precision;
        using Base::stat_record;

        /// \brief Tolerance for feasibility checks
        Scalar tol_feas{1e-8};

        /// \brief Tolerance for KKT conditions
        Scalar tol_ktratio{1e-8};

        /// \brief Whether to enable verbose output
        bool verbose{false};
    };

    namespace details
    {
        template<template<typename> class SolverTpl, typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorXConstraintSolverTpl
        {
        };

        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorXConstraintSolverTpl<::simplex::ADMMContactSolverTpl, _Scalar, _Options, JointCollectionTpl>
        {
            using Scalar = _Scalar;
            enum
            {
                Options = _Options
            };
            using SimulatorX = SimulatorXTpl<Scalar, Options, JointCollectionTpl>;
            using SimulatorState = typename SimulatorX::SimulatorState;
            using ConstraintsProblemDerivatives = typename SimulatorX::ConstraintsProblemDerivatives;
            using ADMMConstraintSolver = typename SimulatorX::ADMMConstraintSolver;
            using ADMMConstraintSolverConfig = typename SimulatorX::ADMMConstraintSolverConfig;

            using MatrixXs = typename SimulatorX::MatrixXs;
            using VectorXs = typename SimulatorX::VectorXs;
            using RefConstVectorXs = Eigen::Ref<const VectorXs>;
            using DelassusType = typename ConstraintsProblemDerivatives::DelassusType;

            static void run(SimulatorX & simulator);

            SIMPLEX_PROTECTED
            static void setup(SimulatorX & simulator);
        };

        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorXConstraintSolverTpl<::pinocchio::PGSContactSolverTpl, _Scalar, _Options, JointCollectionTpl>
        {
            using Scalar = _Scalar;
            enum
            {
                Options = _Options
            };
            using SimulatorX = SimulatorXTpl<Scalar, Options, JointCollectionTpl>;
            using SimulatorState = typename SimulatorX::SimulatorState;
            using ConstraintsProblemDerivatives = typename SimulatorX::ConstraintsProblemDerivatives;
            using PGSConstraintSolver = typename SimulatorX::PGSConstraintSolver;
            using PGSConstraintSolverConfig = typename SimulatorX::PGSConstraintSolverConfig;
            using DelassusOperatorDense = ::pinocchio::DelassusOperatorDenseTpl<Scalar>;

            using MatrixXs = typename SimulatorX::MatrixXs;
            using VectorXs = typename SimulatorX::VectorXs;
            using RefConstVectorXs = Eigen::Ref<const VectorXs>;
            using DelassusType = typename ConstraintsProblemDerivatives::DelassusType;

            static void run(SimulatorX & simulator);

            SIMPLEX_PROTECTED
            static void setup(SimulatorX & simulator);
        };

#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorXConstraintSolverTpl<::simplex::ClarabelContactSolverTpl, _Scalar, _Options, JointCollectionTpl>
        {
            using Scalar = _Scalar;
            enum
            {
                Options = _Options
            };
            using SimulatorX = SimulatorXTpl<Scalar, Options, JointCollectionTpl>;
            using SimulatorState = typename SimulatorX::SimulatorState;
            using ConstraintsProblemDerivatives = typename SimulatorX::ConstraintsProblemDerivatives;
            using ClarabelConstraintSolver = ::simplex::ClarabelContactSolverTpl<Scalar, Options>;
            using ClarabelConstraintSolverConfig = typename SimulatorX::ClarabelConstraintSolverConfig;

            using MatrixXs = typename SimulatorX::MatrixXs;
            using VectorXs = typename SimulatorX::VectorXs;
            using RefConstVectorXs = Eigen::Ref<const VectorXs>;
            using DelassusType = typename ConstraintsProblemDerivatives::DelassusType;

            static void run(SimulatorX & simulator);

            SIMPLEX_PROTECTED
            static void setup(SimulatorX & simulator);
        };
#endif // SIMPLEX_WITH_CLARABEL_SUPPORT

    } // namespace details
} // namespace simplex

#include "simplex/core/simulator-x.hxx"

#if SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION
    #include "simplex/core/simulator-x.txx"
    #include "simplex/pinocchio_template_instantiation/aba.txx"
    #include "simplex/pinocchio_template_instantiation/joint-model.txx"
    #include "simplex/pinocchio_template_instantiation/crba.txx"
#endif

#endif // ifndef __simplex_core_simulator_x_hpp__