#ifndef __simplex_core_simulator_hpp__
#define __simplex_core_simulator_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/core/constraints-problem.hpp"
#include "simplex/macros.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>

#include <pinocchio/algorithm/admm-solver.hpp>
#include <pinocchio/algorithm/pgs-solver.hpp>
#include <pinocchio/algorithm/proximal.hpp>

#include <pinocchio/collision/broadphase-manager.hpp>
#include <coal/broadphase/broadphase_dynamic_AABB_tree.h>

namespace simplex
{
    /**
     * @brief Traits to define associated types for the Simulator.
     */
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct traits<SimulatorTpl<_Scalar, _Options, JointCollectionTpl>>
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
    struct SimulatorTpl
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // --- Type Definitions ---
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
        using CollisionCallbackCollect = ::pinocchio::CollisionCallBackCollect;

        // Contact solvers provided by Pinocchio
        using ADMMConstraintSolver = ::pinocchio::ADMMContactSolverTpl<Scalar>;
        using PGSConstraintSolver = ::pinocchio::PGSContactSolverTpl<Scalar>;

        using SpatialForce = ::pinocchio::ForceTpl<Scalar, Options>;
        using SpatialForceVector = PINOCCHIO_ALIGNED_STD_VECTOR(SpatialForce);

        using ConstraintsProblem = ConstraintsProblemTpl<Scalar, Options, JointCollectionTpl>;
        using ConstraintsProblemHandle = std::shared_ptr<ConstraintsProblem>;

        using ConstraintCholeskyDecomposition = typename ConstraintsProblem::ConstraintCholeskyDecomposition;
        using DelassusOperator = ::pinocchio::DelassusCholeskyExpressionTpl<ConstraintCholeskyDecomposition>;

        // Specific constraint models
        using FrictionalPointConstraintModel = typename ConstraintsProblem::FrictionalPointConstraintModel;
        using FrictionalJointConstraintModel = typename ConstraintsProblem::FrictionalJointConstraintModel;
        using JointLimitConstraintModel = typename ConstraintsProblem::JointLimitConstraintModel;
        using BilateralPointConstraintModel = typename ConstraintsProblem::BilateralPointConstraintModel;
        using BilateralPointConstraintData = typename ConstraintsProblem::BilateralPointConstraintData;
        using BilateralPointConstraintModelVector = typename ConstraintsProblem::BilateralPointConstraintModelVector;
        using WeldConstraintModel = typename ConstraintsProblem::WeldConstraintModel;
        using WeldConstraintData = typename ConstraintsProblem::WeldConstraintData;
        using WeldConstraintModelVector = typename ConstraintsProblem::WeldConstraintModelVector;

        using ConstraintModel = typename ConstraintsProblem::ConstraintModel;
        using ConstraintData = typename ConstraintsProblem::ConstraintData;

        SIMPLEX_PROTECTED
        /// \brief Handle to the model of the system.
        ModelHandle m_model;

        /// \brief Handle to the model's data of the system.
        DataHandle m_data;

        /// \brief Handle to the geometry model of the system.
        GeometryModelHandle m_geom_model;

        /// \brief Handle to the geometry model's data of the system.
        GeometryDataHandle m_geom_data;

        SIMPLEX_PUBLIC
        // --- Simulator State Variables ---
        /// \brief Joints configuration of the system - copied from `step` input
        VectorXs q;

        /// \brief Joints velocity of the system - copied from `step` input
        VectorXs v;

        /// \brief External joint torques - copied from `step` input
        VectorXs tau;

        /// \brief External 6D force exerted in the local frame of each joint - copied from `step` input
        SpatialForceVector fext;

        /// \brief Time step of the simulator - copied from `step` input
        Scalar dt;

        /// \brief The updated joint configuration of the system.
        VectorXs qnew;

        /// \brief Free velocity i.e. the updated velocity of the system without any constraint forces.
        VectorXs vfree;

        /// \brief The updated velocity, taking into account the correction due to constraint forces.
        VectorXs vnew;

        /// \brief The updated acceleration, taking into account the correction due to constraint forces.
        VectorXs anew;

        /// \brief Vector of total spatial forces (i.e. external forces + constraint forces) applied on
        /// joints, expressed in the local frame of each joint. Note: by subtracting the external forces
        /// (given to the `step` method of the simulator) to `ftotal`, we get the constraint forces
        /// expressed in the local frame of the joints.
        SpatialForceVector ftotal;

        /// \brief Vector of total torques (i.e. applied tau + dry friction on joints + forces due to joint limits) applied on
        /// joints.
        VectorXs tau_total;

        /// \brief Vector of constraint torques
        VectorXs tau_constraints;

        // --- Solver Settings ---
        // refer to paper: "From Compliant to Rigid Contact Simulation: a Unified and Efficient Approach"
        struct ConstraintSolverSettings
        {
            /// @brief Maximum number of iterations allowed for the solver (n_iter in Algorithm 1).
            int max_iter{1000};

            /// @brief Absolute convergence tolerance (epsilon_abs in Eq. 43).
            /// The solver stops when primal, dual, and complementarity residuals are below this value.
            Scalar absolute_precision{1e-8};

            /// @brief Relative convergence tolerance.
            /// Used to account for the scale of the problem variables, avoiding stagnation due to numerical limits.
            Scalar relative_precision{1e-8};

            /// @brief Flag to enable or disable the collection of solver statistics (e.g., iteration count, residuals).
            bool stat_record{false};
        };

        /** Settings tailored for ADMM solver (regularization and update rules). */
        /// \brief Struct to store the settings for the ADMM constraint solver.
        ///
        /// Pinocchio's ADMM has two regularization term
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
        struct ADMMConstraintSolverSettings : ConstraintSolverSettings
        {
            /// @brief Proximal regularization term (eta in Eq. 35 and Eq. 37).
            /// This term ensures the linear system in the f-update step is always strictly positive definite (invertible),
            /// even for rigid contacts (R=0) or ill-conditioned Delassus matrices.
            Scalar mu{1e-6}; // η

            /// \brief Linear scaling of the ADMM penalty term
            Scalar tau{0.5};

            /// @brief The Augmented Lagrangian penalty parameter (rho in Eq. 33).
            /// This is the "step size" of the dual update. It balances the convergence speed between
            /// satisfying the constraints (primal residual) and minimizing the objective (dual residual).
            // initial value of rho for linear update rule
            Scalar rho{10}; // ρ

            /// @brief The exponent parameter for the Spectral Update Rule (p in Eq. 45).
            /// Used when admm_update_rule is SPECTRAL. It scales rho based on the condition number (kappa)
            /// of the problem: rho = sqrt(m*L) * kappa^p.
            // initial value of the rho_power for spectral update rule
            Scalar rho_power{0.2};

            /// @brief The increment/decrement step for the rho_power parameter (p_inc / p_dec in Eq. 46).
            /// In the Spectral Update Rule, if residuals are unbalanced, 'p' is adjusted by this factor
            /// to adapt rho dynamically.
            Scalar rho_power_factor{0.05};

            /// @brief The multiplication/division factor for the Linear Update Rule (tau_inc / tau_dec in Eq. 44).
            /// Used when admm_update_rule is LINEAR. If primal/dual residuals diverge, rho is multiplied
            /// or divided by this factor (typically > 1).
            Scalar linear_update_rule_factor{2};

            /// @brief The threshold ratio between primal and dual residuals (alpha in Eq. 44 and Eq. 46).
            /// This defines the "tube" width. If one residual is 'alpha' times larger than the other,
            /// the update rule (Linear or Spectral) is triggered to adjust rho.
            Scalar ratio_primal_dual{50}; // α

            /// @brief Number of iterations for the power iteration or Lanczos algorithm.
            /// Used in the Spectral Update Rule to estimate the largest (L) and smallest (m) eigenvalues
            /// of the augmented Delassus matrix (Eq. 45) to compute the condition number.
            // higher leads to more accurate eigvalues estimation. Max size = constraint problem size
            int lanczos_size{3};

            /// @brief Strategy for updating the penalty parameter rho (Section III-F).
            /// Options:
            /// - LINEAR: Standard ADMM update based on residual ratios (Eq. 44).
            /// - SPECTRAL: Novel update rule based on spectral properties of the Delassus matrix (Eq. 45, 46).
            ::pinocchio::ADMMUpdateRule admm_update_rule{::pinocchio::ADMMUpdateRule::SPECTRAL};
        };

        /// \brief Settings for the ADMM constraint solver.
        ADMMConstraintSolverSettings admm_constraint_solver_settings;

        /// \brief Struct to store the settings for the PGS constraint solver.
        struct PGSConstraintSolverSettings : ConstraintSolverSettings
        {
            Scalar over_relax{1.0};
        };

        /// \brief Settings for the PGS constraint solver.
        PGSConstraintSolverSettings pgs_constraint_solver_settings;

        /// \brief Type of constraint solver used in the last call to the `step` method.
        /// This enum is meant to indicate which solver was used, not which one to use.
        /// To use a specific constraint solver, use the template of the `step` method.
        enum struct ConstraintSolverType
        {
            PGS,
            ADMM,
            NONE
        };

        /// \brief Type of constraint solver used in the last call to the `step` method.
        ConstraintSolverType constraint_solver_type;

        /// \brief ADMM constraint solver.
        ADMMConstraintSolver admm_constraint_solver;

        /// \brief PGS constraint solver.
        PGSConstraintSolver pgs_constraint_solver;

        /// \brief Whether or not to warm-start the constraint solver.
        bool warm_start_constraints_forces{true};

        /// \brief Whether or not timings of the `step` function are measured.
        /// If set to true, the timing of the last call to `step` can be accessed via `getCPUTimes`
        bool measure_timings{false};

        // --- Timing Utilities ---
        /// \brief Get timings of the last call to the `step` method.
        coal::CPUTimes getStepCPUTimes() const
        {
            return this->m_step_timings;
        }

        /// \brief Get timings of the call to the constraint solver in the last call of `step`.
        /// This timing is set to 0 if there was no constraints.
        coal::CPUTimes getConstraintSolverCPUTimes() const
        {
            return this->m_constraint_solver_timings;
        }

        /// \brief Get timings of the collision detection stage.
        coal::CPUTimes getCollisionDetectionCPUTimes() const
        {
            return this->m_collision_detection_timings;
        }

        /// \brief Virtual destructor of the base class
        virtual ~SimulatorTpl()
        {
        }

        SIMPLEX_PROTECTED
        /// \brief Broad phase manager
        // TODO: template by broad phase type (no broad phase, dynamic aabb tree, SAP etc)
        BroadPhaseManagerHandle m_broad_phase_manager;

        /// \brief Collision callback
        CollisionCallbackCollect m_collision_callback;

        /// \brief Constraint problem
        ConstraintsProblemHandle m_constraints_problem;

        /// \brief Temporary variable used to integrate the system's state
        VectorXs m_vnew_integration_tmp;

        /// \brief Whether or not the simulator has been reset.
        bool m_is_reset;

        // Profiling timers
        /// \brief Timer used for the whole `step` method
        coal::Timer m_step_timer;

        /// \brief Timings for the whole `step` method.
        coal::CPUTimes m_step_timings;

        /// \brief Timer used for internal methods inside `step`.
        coal::Timer m_internal_timer;

        /// \brief Timings for the call to the constraint solver.
        coal::CPUTimes m_constraint_solver_timings;

        /// \brief Timings for collision detection.
        coal::CPUTimes m_collision_detection_timings;

        SIMPLEX_PUBLIC
        // --- Constructors ---
        /// \brief Constructor specifying the Data and GeometryData associated to the model and geom_model.
        SimulatorTpl(
            ModelHandle model_handle,
            DataHandle data_handle,
            GeometryModelHandle geom_model_handle,
            GeometryDataHandle geom_data_handle,
            const BilateralPointConstraintModelVector & bilateral_constraint_models,
            const WeldConstraintModelVector & weld_constraint_models);

        /// \brief Constructor using model, data, geometry model, geometry data and vector of bilateral constraints.
        SimulatorTpl(
            ModelHandle model_handle,
            DataHandle data_handle,
            GeometryModelHandle geom_model_handle,
            GeometryDataHandle geom_data_handle,
            const BilateralPointConstraintModelVector & bilateral_constraint_models);

        /// \brief Constructor using model, data, geometry model, geometry data and vector of weld constraints.
        SimulatorTpl(
            ModelHandle model_handle,
            DataHandle data_handle,
            GeometryModelHandle geom_model_handle,
            GeometryDataHandle geom_data_handle,
            const WeldConstraintModelVector & weld_constraint_models);

        /// \brief Default constructor
        /// Whenever the model or the geometry model is changed, this constructor should be called.
        SimulatorTpl(
            ModelHandle model_handle, DataHandle data_handle, GeometryModelHandle geom_model_handle, GeometryDataHandle geom_data_handle);

        /// \brief Default constructor
        /// Whenever the model or the geometry model is changed, this constructor should be called.
        SimulatorTpl(
            ModelHandle model_handle,
            GeometryModelHandle geom_model_handle,
            const BilateralPointConstraintModelVector & bilateral_constraint_models);

        /// \brief Default constructor
        /// Whenever the model or the geometry model is changed, this constructor should be called.
        SimulatorTpl(
            ModelHandle model_handle, GeometryModelHandle geom_model_handle, const WeldConstraintModelVector & weld_constraint_models);

        /// \brief Default constructor
        /// Whenever the model or the geometry model is changed, this constructor should be called.
        SimulatorTpl(
            ModelHandle model_handle,
            GeometryModelHandle geom_model_handle,
            const BilateralPointConstraintModelVector & bilateral_constraint_models,
            const WeldConstraintModelVector & weld_constraint_models);

        /// \brief Default constructor
        /// Whenever the model or the geometry model is changed, this constructor should be called.
        SimulatorTpl(ModelHandle model_handle, GeometryModelHandle geom_model_handle);

        /// \brief Resets the internal quantities of the simulator.
        /// This methods needs to be called before looping on the `step` method, for example when the
        /// initial state (q0, v0) of the system is used as an input to `step` and (q0, v0) have not
        /// been computed using the `step` method. \note If, instead, the simulator's model, data,
        /// geom_model or geom_data have changed, please call the constructor.
        void reset();

        /** @brief Primary simulation step function. Solves constraints and integrates dynamics. */
        template<
            template<typename> class ConstraintSolver = ::pinocchio::ADMMContactSolverTpl,
            typename ConfigVectorType,
            typename VelocityVectorType,
            typename TorqueVectorType>
        void step(
            const Eigen::MatrixBase<ConfigVectorType> & q,
            const Eigen::MatrixBase<VelocityVectorType> & v,
            const Eigen::MatrixBase<TorqueVectorType> & tau,
            Scalar dt);

        /** @brief Overloaded step function with external spatial forces support. fext must be expressed in the local frame of each joint.
            TODO: remove aligned_vector and template by Allocator.
            TODO: change MatrixBase to PlainObject
        **/
        template<
            template<typename> class ConstraintSolver = ::pinocchio::ADMMContactSolverTpl,
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

        /// \brief Returns true if the simulator is in its reset state.
        bool isReset() const
        {
            return this->m_is_reset;
        }

        /// \brief Check consistency of the simulator.
        bool check() const;

        /// \brief Check consistency of the collision pairs.
        /// This method checks thath the geometry model does not contain collision pairs between
        /// geometry objects from the same parent joint.
        bool checkCollisionPairs() const;

        // --- Accessors ---
        /// \brief Returns a const reference to the model
        const Model & model() const
        {
            return pinocchio::helper::get_ref(this->m_model);
        }

        /// \brief Returns a reference to the model
        Model & model()
        {
            return pinocchio::helper::get_ref(this->m_model);
        }

        /// \brief Returns a const reference to the data
        const Data & data() const
        {
            return pinocchio::helper::get_ref(this->m_data);
        }

        /// \brief Returns a reference to the data
        Data & data()
        {
            return pinocchio::helper::get_ref(this->m_data);
        }

        /// \brief Returns a const reference to the geometry model
        const pinocchio::GeometryModel & geom_model() const
        {
            return pinocchio::helper::get_ref(this->m_geom_model);
        }

        /// \brief Returns a reference to the geometry model
        pinocchio::GeometryModel & geom_model()
        {
            return pinocchio::helper::get_ref(this->m_geom_model);
        }

        /// \brief Returns a const reference to the geometry data
        const pinocchio::GeometryData & geom_data() const
        {
            return pinocchio::helper::get_ref(this->m_geom_data);
        }

        /// \brief Returns a reference to the geometry data
        pinocchio::GeometryData & geom_data()
        {
            return pinocchio::helper::get_ref(this->m_geom_data);
        }

        /// \brief Returns a const reference to the constraint problem
        const ConstraintsProblem & constraints_problem() const
        {
            return pinocchio::helper::get_ref(this->m_constraints_problem);
        }

        /// \brief Returns a reference to the constraint problem
        ConstraintsProblem & constraints_problem()
        {
            return pinocchio::helper::get_ref(this->m_constraints_problem);
        }

        /// \brief Returns a handle to the constraint problem
        ConstraintsProblemHandle getConstraintsProblemHandle() const
        {
            return this->m_constraints_problem;
        }

        SIMPLEX_PROTECTED
        /// \brief Allocates memory based on `model` and active collision pairs in `geom_model`.
        void allocate();

        /// \brief Initializes the geometry data for broad and narrow phase collision detection.
        void initializeGeometryData();

        /// \brief Warm starting constraint forces via constraint inverse dynamics.
        void warmStartConstraintForces();

        /// \brief Collision detection
        void detectCollisions();

        /// \brief Constraint resolution
        virtual void preambleResolveConstraints(const Scalar dt)
        {
            PINOCCHIO_UNUSED_VARIABLE(dt);
        }

        /** @brief Internal method to resolve numerical constraints using a specific solver. */
        template<template<typename> class ConstraintSolver, typename VelocityVectorType>
        void resolveConstraints(const Eigen::MatrixBase<VelocityVectorType> & v, const Scalar dt);
    };

    namespace details
    {
        /** @brief Internal dispatcher for the simulator's constraint solver. */
        template<template<typename> class SolverTpl, typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorConstraintSolverTpl
        {
        };

        /** @brief ADMM specialization of the dispatcher. */
        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorConstraintSolverTpl<::pinocchio::ADMMContactSolverTpl, _Scalar, _Options, JointCollectionTpl>
        {
            using Scalar = _Scalar;
            enum
            {
                Options = _Options
            };
            using Simulator = SimulatorTpl<Scalar, Options, JointCollectionTpl>;
            using ConstraintsProblem = typename Simulator::ConstraintsProblem;
            using ADMMConstraintSolverSettings = typename Simulator::ADMMConstraintSolverSettings;
            using ADMMConstraintSolver = ::pinocchio::ADMMContactSolverTpl<Scalar>;

            using MatrixXs = typename Simulator::MatrixXs;
            using VectorXs = typename Simulator::VectorXs;
            using RefConstVectorXs = Eigen::Ref<const VectorXs>;

            static void run(Simulator & simulator, Scalar dt);

            SIMPLEX_PROTECTED
            static void setup(Simulator & simulator);
        };

        /** @brief PGS specialization of the dispatcher. */
        template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
        struct SimulatorConstraintSolverTpl<::pinocchio::PGSContactSolverTpl, _Scalar, _Options, JointCollectionTpl>
        {
            using Scalar = _Scalar;
            enum
            {
                Options = _Options
            };
            using Simulator = SimulatorTpl<Scalar, Options, JointCollectionTpl>;
            using ConstraintsProblem = typename Simulator::ConstraintsProblem;
            using PGSConstraintSolverSettings = typename Simulator::PGSConstraintSolverSettings;
            using PGSConstraintSolver = ::pinocchio::PGSContactSolverTpl<Scalar>;
            using DelassusOperatorDense = ::pinocchio::DelassusOperatorDenseTpl<Scalar>;

            using MatrixXs = typename Simulator::MatrixXs;
            using VectorXs = typename Simulator::VectorXs;
            using RefConstVectorXs = Eigen::Ref<const VectorXs>;

            static void run(Simulator & simulator, Scalar dt);

            SIMPLEX_PROTECTED
            static void setup(Simulator & simulator);
        };
    } // namespace details
} // namespace simplex

#include "simplex/core/simulator.hxx"

#if SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION
    #include "simplex/core/simulator.txx"
    #include "simplex/pinocchio_template_instantiation/aba.txx"
    #include "simplex/pinocchio_template_instantiation/joint-model.txx"
    #include "simplex/pinocchio_template_instantiation/crba.txx"
#endif

#endif // ifndef __simplex_core_simulator_hpp__