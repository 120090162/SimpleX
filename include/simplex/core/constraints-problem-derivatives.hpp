#ifndef __simplex_core_constraints_problem_derivatives_hpp__
#define __simplex_core_constraints_problem_derivatives_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/macros.hpp"
#include "simplex/core/constraints-problem.hpp"
#include "simplex/utils/visitors.hpp"

#include <pinocchio/container/storage.hpp>

namespace simplex
{

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct traits<ConstraintsProblemDerivativesTpl<_Scalar, _Options, JointCollectionTpl>>
    {
        using Scalar = _Scalar;
    };

    /**
     * @brief Extension of the standard ConstraintsProblem to handle derivatives computation.
     *
     * Introduces the optimized storage layout to handle the Jacobian of the
     * constraints (dg/dq, dg/dv) required for Differentiable Simulation (CIMPC).
     */
    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct ConstraintsProblemDerivativesTpl : ConstraintsProblemTpl<_Scalar, _Options, JointCollectionTpl>
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // -------------------------------------------------------------------------------------------------
        // TYPEDEFS
        // -------------------------------------------------------------------------------------------------
        using Base = ConstraintsProblemTpl<_Scalar, _Options, JointCollectionTpl>;
        using Scalar = typename Base::Scalar;
        enum
        {
            Options = _Options
        };

        using GeomIndex = typename Base::GeomIndex;
        using JointIndex = typename Base::JointIndex;
        using ContactIndex = typename Base::ContactIndex;

        using VectorXs = typename Base::VectorXs;
        using MatrixXs = typename Base::MatrixXs;
        using Vector3s = typename Base::Vector3s;
        using Vector6s = typename Base::Vector6s;
        using MapVectorXs = typename Base::MapVectorXs;

        using VectorStorage = ::pinocchio::EigenStorageTpl<VectorXs>;
        using VectorStorageMapType = typename VectorStorage::MapType;
        using VectorStorageRefMapType = typename VectorStorage::RefMapType;
        using VectorView = Eigen::VectorBlock<VectorStorageMapType>;
        using MatrixStorage = ::pinocchio::EigenStorageTpl<MatrixXs>;
        using MatrixStorageMapType = typename MatrixStorage::MapType;
        using MatrixStorageRefMapType = typename MatrixStorage::RefMapType;
        using VectorIndex = Eigen::DenseIndex;

        using ModelHandle = typename Base::ModelHandle;
        using DataHandle = typename Base::DataHandle;
        using GeometryModelHandle = typename Base::GeometryModelHandle;
        using GeometryDataHandle = typename Base::GeometryDataHandle;

        using ConstraintModel = typename Base::ConstraintModel;
        using ConstraintData = typename Base::ConstraintData;

        using FrictionalJointConstraintModel = typename Base::FrictionalJointConstraintModel;
        using FrictionalJointConstraintData = typename Base::FrictionalJointConstraintData;

        using BilateralPointConstraintModel = typename Base::BilateralPointConstraintModel;
        using BilateralPointConstraintData = typename Base::BilateralPointConstraintData;
        using BilateralPointConstraintModelVector = typename Base::BilateralPointConstraintModelVector;

        using WeldConstraintModel = typename Base::WeldConstraintModel;
        using WeldConstraintData = typename Base::WeldConstraintData;
        using WeldConstraintModelVector = typename Base::WeldConstraintModelVector;

        using JointLimitConstraintModel = typename Base::JointLimitConstraintModel;
        using JointLimitConstraintData = typename Base::JointLimitConstraintData;

        using FrictionalPointConstraintModel = typename Base::FrictionalPointConstraintModel;
        using FrictionalPointConstraintData = typename Base::FrictionalPointConstraintData;

        using CoulombFrictionCone = typename Base::CoulombFrictionCone;

        using SE3 = typename Base::SE3;
        using BoxSet = typename Base::BoxSet;
        using PlacementVector = typename Base::PlacementVector;
        using ContactMapper = typename Base::ContactMapper;

        // -------------------------------------------------------------------------------------------------
        // MEMBERS
        // -------------------------------------------------------------------------------------------------
        SIMPLEX_PUBLIC
        /// \brief Type of the delassus:
        /// - cholesky (needs to allocate memory to store the cholesky decomposition)
        /// - rigid body (uses the kinematic tree of the scene, memory-allocation free)
        /// \note If CHOLESKY is selected, then the decomposition is computed in the `build` method.
        enum struct DelassusType
        {
            DENSE,
            CHOLESKY,
            RIGID_BODY,
        } delassus_type{DelassusType::CHOLESKY};

        /// \brief Resizable storage for dynamics quantities related to constraints.
        struct ConstraintProblemStorage
        {
            /// \brief Resizable memory to store the delassus matrix if needed.
            MatrixStorage delassus_matrix_storage;

            /// \brief Storage for the drift term of the constraints problem.
            VectorStorage g_storage;

            /// \brief Storage for constraint forces.
            VectorStorage constraint_forces_storage;

            /// \brief Storage for constraint velocities.
            VectorStorage constraint_velocities_storage;

            /// \brief Storage for the preconditionner of the constraints problem.
            VectorStorage preconditioner_storage;

            /// \brief Storage for time scaling factors to convert acceleration units to the units of each constraint.
            VectorStorage time_scaling_acc_to_constraints_storage;

            ///
            /// \brief Reserve memory for the constraints problem storage.
            void reserve(const int max_constraint_problem_size)
            {
                assert(max_constraint_problem_size >= 0);
                // Don't reserve maximum storage for delassus as it can be huge.
                // delassus_matrix_storage.reserve(max_constraint_problem_size, max_constraint_problem_size);
                g_storage.reserve(max_constraint_problem_size);
                time_scaling_acc_to_constraints_storage.reserve(max_constraint_problem_size);
                constraint_forces_storage.reserve(max_constraint_problem_size);
                constraint_velocities_storage.reserve(max_constraint_problem_size);
                preconditioner_storage.reserve(max_constraint_problem_size);
                time_scaling_acc_to_constraints_storage.reserve(max_constraint_problem_size);
            }

            ///
            /// \brief Resizes the maps of each vector storage to the constraint problem's size.
            void resize(const int constraint_problem_size)
            {
                assert(constraint_problem_size >= 0);
                delassus_matrix_storage.resize(constraint_problem_size, constraint_problem_size);
                g_storage.resize(constraint_problem_size);
                constraint_forces_storage.resize(constraint_problem_size);
                constraint_velocities_storage.resize(constraint_problem_size);
                preconditioner_storage.resize(constraint_problem_size);
                time_scaling_acc_to_constraints_storage.resize(constraint_problem_size);
            }
        } storage;

        /// \brief Map to the delassus matrix.
        MatrixStorageRefMapType delassus_matrix;

        /// \brief Drift term of the constraints problem.
        VectorStorageRefMapType g;

        /// \brief Storage for constraint forces.
        VectorStorageRefMapType constraint_forces;

        /// \brief Storage for constraint velocities.
        VectorStorageRefMapType constraint_velocities;

        /// \brief Drift term of the constraints problem.
        VectorStorageRefMapType preconditioner;

        /// \brief Storage for time scaling factors to convert acceleration units to the units of each constraint.
        /// TODO(louis): remove
        VectorStorageRefMapType time_scaling_acc_to_constraints;

        /// ----------------------------------
        /// Joints dry frictions constraint
        /// \brief Lower bound on joint friction constraint forces (in Newton)
        VectorXs joint_friction_lower_limit;

        /// \brief Upper bound on joint friction constraint forces (in Newton)
        VectorXs joint_friction_upper_limit;
        /// ----------------------------------

        /// \brief Contact placements.
        PlacementVector point_contact_constraint_placements;

        // -------------------------------------------------------------------------------------------------
        // CONSTRUCTORS
        // -------------------------------------------------------------------------------------------------
        SIMPLEX_PUBLIC
        ///
        /// \brief Default constructor.
        ConstraintsProblemDerivativesTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle,
            const BilateralPointConstraintModelVector & bilateral_point_constraint_models,
            const WeldConstraintModelVector & weld_constraint_models)
        : Base(
              model_handle,
              data_handle,
              geom_model_handle,
              geom_data_handle,
              bilateral_point_constraint_models,
              weld_constraint_models,
              false)
        , delassus_matrix(storage.delassus_matrix_storage.map())
        , g(storage.g_storage.map())
        , constraint_forces(storage.constraint_forces_storage.map())
        , constraint_velocities(storage.constraint_velocities_storage.map())
        , preconditioner(storage.preconditioner_storage.map())
        , time_scaling_acc_to_constraints(storage.time_scaling_acc_to_constraints_storage.map())
        {
            // Initial allocation
            this->allocate();
        }

        /// \brief Default constructor.
        ConstraintsProblemDerivativesTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle,
            const BilateralPointConstraintModelVector & bilateral_point_constraint_models)
        : ConstraintsProblemDerivativesTpl(
              model_handle,                      //
              data_handle,                       //
              geom_model_handle,                 //
              geom_data_handle,                  //
              bilateral_point_constraint_models, //
              WeldConstraintModelVector())
        {
        }

        /// \brief Default constructor.
        ConstraintsProblemDerivativesTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle,
            const WeldConstraintModelVector & weld_constraint_models)
        : ConstraintsProblemDerivativesTpl(
              model_handle,                          //
              data_handle,                           //
              geom_model_handle,                     //
              geom_data_handle,                      //
              BilateralPointConstraintModelVector(), //
              weld_constraint_models)
        {
        }

        /// \brief Default constructor.
        ConstraintsProblemDerivativesTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle)
        : ConstraintsProblemDerivativesTpl(
              model_handle,                          //
              data_handle,                           //
              geom_model_handle,                     //
              geom_data_handle,                      //
              BilateralPointConstraintModelVector(), //
              WeldConstraintModelVector())
        {
        }

        // -------------------------------------------------------------------------------------------------
        // OVERLOADED METHODS
        // -------------------------------------------------------------------------------------------------
        SIMPLEX_PROTECTED
        ///
        /// \brief Compute constraint drift g = Jc * vfree + baumgarte.
        template<typename FreeVelocityVectorType, typename VelocityVectorType>
        void computeConstraintsDrift(
            const Eigen::MatrixBase<FreeVelocityVectorType> & vfree, const Eigen::MatrixBase<VelocityVectorType> & v, const Scalar dt);
        SIMPLEX_PUBLIC
        /// \brief Allocates memory for the constraints problem quantities.
        /// Notes:
        ///   - This method uses the the geometry model's active collision pairs to allocate memory.
        ///   - because we always resize the constraints problem quantities, there won't be any error if
        ///   this method is not called.
        ///     This method is meant to optimize memory allocation for advanced users.
        void allocate();

        /// \brief Empties constraints problem quantities.
        void clear();

        ///
        /// \brief After `model`, `data`, `geom_model` and `geom_data` have been updated, this function updates `constraints`.
        void update(const Scalar dt);

        /// \brief Build the constraints problem quantities: `G`, `g`.
        /// Also builds the quantities necessary to warm-start the constraint solver.
        /// Meant to be called after `update`.
        template<typename FreeVelocityVectorType, typename VelocityVectorType>
        void build(const Eigen::MatrixBase<FreeVelocityVectorType> & vfree, const Eigen::MatrixBase<VelocityVectorType> & v, Scalar dt);

        /// \brief Checks consistency of the constraints problem w.r.t to its handles.
        bool check() const;

        /// \brief Collecting active set from the solution of the contact problem.
        /// the contact problem should be solved before calling this method.
        void collectActiveSet(Scalar epsilon = 1e-6);

        /// \brief Getter for forces and velocities of joints friction constraint.
        VectorView joint_friction_constraint_forces();
        VectorView joint_friction_constraint_forces() const;
        VectorView joint_friction_constraint_velocities();
        VectorView joint_friction_constraint_velocities() const;

        /// \brief Getter for bilateral constraints' forces and velocities.
        VectorView bilateral_constraints_forces();
        VectorView bilateral_constraints_forces() const;
        VectorView bilateral_constraints_velocities();
        VectorView bilateral_constraints_velocities() const;

        /// \brief Getter for weld constraints' forces and velocities.
        VectorView weld_constraints_forces();
        VectorView weld_constraints_forces() const;
        VectorView weld_constraints_velocities();
        VectorView weld_constraints_velocities() const;

        /// \brief Getter for forces and velocities of joint constraints.
        VectorView joint_limit_constraint_forces();
        VectorView joint_limit_constraint_forces() const;
        VectorView joint_limit_constraint_velocities();
        VectorView joint_limit_constraint_velocities() const;

        /// \brief Getter for forces and velocities of frictional point contact constraints.
        VectorView frictional_point_constraints_forces();
        VectorView frictional_point_constraints_forces() const;
        VectorView frictional_point_constraints_velocities();
        VectorView frictional_point_constraints_velocities() const;

        /// \brief Getter for the time scaling factors to convert acceleration units to the units of each constraint.
        VectorView contact_time_scaling_acc_to_constraints()
        {
            VectorIndex begin = this->joint_friction_constraint_size() //
                                + this->bilateral_constraints_size()   //
                                + this->weld_constraints_size()        //
                                + this->joint_limit_constraint_size();
            VectorIndex size = this->frictional_point_constraints_size();
            assert(begin >= 0);
            assert(size >= 0);
            assert(begin + size <= time_scaling_acc_to_constraints.size());
            return time_scaling_acc_to_constraints.segment(begin, size);
        }

        /// \brief Const getter for the time scaling factors to convert acceleration units to the units of each constraint.
        VectorView contact_time_scaling_acc_to_constraints() const
        {
            VectorIndex begin = this->joint_friction_constraint_size() //
                                + this->bilateral_constraints_size()   //
                                + this->weld_constraints_size()        //
                                + this->joint_limit_constraint_size();
            VectorIndex size = this->frictional_point_constraints_size();
            assert(begin >= 0);
            assert(size >= 0);
            assert(begin + size <= time_scaling_acc_to_constraints.size());
            return time_scaling_acc_to_constraints.segment(begin, size);
        }
    };

    // ---------------------------------------------------------------------------
    // ----------------------------- IMPLEMENTATIONS -----------------------------
    // ---------------------------------------------------------------------------

#define CONSTRAINT_FORCES_VELOCITIES_GETTER(constraint_forces_name, constraint_velocities_name, cbegin, csize)                             \
    template<typename S, int O, template<typename, int> class JointCollectionTpl>                                                          \
    typename ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::VectorView                                                        \
    ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::ConstraintsProblemDerivativesTpl::constraint_forces_name()                 \
    {                                                                                                                                      \
        assert(cbegin >= 0);                                                                                                               \
        assert(csize >= 0);                                                                                                                \
        assert(cbegin + csize <= constraint_forces.size());                                                                                \
        return constraint_forces.segment(cbegin, csize);                                                                                   \
    }                                                                                                                                      \
    template<typename S, int O, template<typename, int> class JointCollectionTpl>                                                          \
    typename ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::VectorView                                                        \
    ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::ConstraintsProblemDerivativesTpl::constraint_forces_name() const           \
    {                                                                                                                                      \
        assert(cbegin >= 0);                                                                                                               \
        assert(csize >= 0);                                                                                                                \
        assert(cbegin + csize <= constraint_forces.size());                                                                                \
        return constraint_forces.segment(cbegin, csize);                                                                                   \
    }                                                                                                                                      \
    template<typename S, int O, template<typename, int> class JointCollectionTpl>                                                          \
    typename ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::VectorView                                                        \
    ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::ConstraintsProblemDerivativesTpl::constraint_velocities_name()             \
    {                                                                                                                                      \
        assert(cbegin >= 0);                                                                                                               \
        assert(csize >= 0);                                                                                                                \
        assert(cbegin + csize <= constraint_velocities.size());                                                                            \
        return constraint_velocities.segment(cbegin, csize);                                                                               \
    }                                                                                                                                      \
    template<typename S, int O, template<typename, int> class JointCollectionTpl>                                                          \
    typename ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::VectorView                                                        \
    ConstraintsProblemDerivativesTpl<S, O, JointCollectionTpl>::ConstraintsProblemDerivativesTpl::constraint_velocities_name() const       \
    {                                                                                                                                      \
        assert(cbegin >= 0);                                                                                                               \
        assert(csize >= 0);                                                                                                                \
        assert(cbegin + csize <= constraint_velocities.size());                                                                            \
        return constraint_velocities.segment(cbegin, csize);                                                                               \
    }

    CONSTRAINT_FORCES_VELOCITIES_GETTER(
        joint_friction_constraint_forces,      // forces getter
        joint_friction_constraint_velocities,  // velocities getter
        0,                                     // storage begin index
        this->joint_friction_constraint_size() // constraint size
    );

    CONSTRAINT_FORCES_VELOCITIES_GETTER(
        bilateral_constraints_forces, //
        bilateral_constraints_velocities,
        this->joint_friction_constraint_size(),
        this->bilateral_constraints_size());

    CONSTRAINT_FORCES_VELOCITIES_GETTER(
        weld_constraints_forces,
        weld_constraints_velocities,
        this->joint_friction_constraint_size() + this->bilateral_constraints_size(),
        this->weld_constraints_size());

    CONSTRAINT_FORCES_VELOCITIES_GETTER(
        joint_limit_constraint_forces,
        joint_limit_constraint_velocities,
        this->joint_friction_constraint_size() + this->bilateral_constraints_size() + this->weld_constraints_size(),
        this->joint_limit_constraint_size());

    CONSTRAINT_FORCES_VELOCITIES_GETTER(
        frictional_point_constraints_forces,
        frictional_point_constraints_velocities,
        this->joint_friction_constraint_size()     //
            + this->bilateral_constraints_size()   //
            + this->weld_constraints_size()        //
            + this->joint_limit_constraint_size(), // storage begin index
        this->frictional_point_constraints_size()  // constraint size
    );

#undef CONSTRAINT_FORCES_VELOCITIES_GETTER

} // namespace simplex

/* --- Details -------------------------------------------------------------- */
#include "simplex/core/constraints-problem-derivatives.hxx"

#if SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION
    #include "simplex/core/constraints-problem-derivatives.txx"
#endif

#endif // __simplex_core_constraints_problem_derivatives_hpp__