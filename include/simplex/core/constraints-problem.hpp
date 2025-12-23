#ifndef __simplex_core_constraints_problem_hpp__
#define __simplex_core_constraints_problem_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/core/contact-frame.hpp"

#include <pinocchio/algorithm/delassus-operator-base.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/constraints/coulomb-friction-cone.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>

namespace simplex
{

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct traits<ConstraintsProblemTpl<_Scalar, _Options, JointCollectionTpl>>
    {
        using Scalar = _Scalar;
    };

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct ConstraintsProblemTpl // : ::pinocchio::NumericalBase<ConstraintsProblemTpl<_Scalar, _Options,
                                 // JointCollectionTpl>>
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        /////////////////////////////////////////////////
        /// TYPEDEFS
        /////////////////////////////////////////////////
        using Scalar = _Scalar;
        enum
        {
            Options = _Options
        };

        using GeomIndex = pinocchio::GeomIndex;
        using JointIndex = pinocchio::JointIndex;

        using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
        using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;
        using Vector3s = Eigen::Matrix<Scalar, 3, 1, Options>;
        using Vector6s = Eigen::Matrix<Scalar, 6, 1, Options>;
        using Matrix3s = Eigen::Matrix<Scalar, 3, 3, Options>;
        using MapVectorXs = Eigen::Map<VectorXs>;
        using MapVector3s = Eigen::Map<Vector3s>;
        using MapVector6s = Eigen::Map<Vector6s>;
        using ContactIndex = std::size_t;

        using Model = ::pinocchio::ModelTpl<Scalar, Options, JointCollectionTpl>;
        using ModelHandle = std::shared_ptr<Model>;
        using Data = typename Model::Data;
        using DataHandle = std::shared_ptr<Data>;
        using GeometryModelHandle = std::shared_ptr<::pinocchio::GeometryModel>;
        using GeometryDataHandle = std::shared_ptr<::pinocchio::GeometryData>;

        using ConstraintModel = ::pinocchio::ConstraintModelTpl<Scalar, Options>;
        using ConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(ConstraintModel);
        using WrappedConstraintModel = std::reference_wrapper<const ConstraintModel>;
        using WrappedConstraintModelVector = std::vector<WrappedConstraintModel>;

        using ConstraintData = ::pinocchio::ConstraintDataTpl<Scalar, Options>;
        using ConstraintDataVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(ConstraintData);
        using WrappedConstraintData = std::reference_wrapper<ConstraintData>;
        using WrappedConstraintDataVector = std::vector<WrappedConstraintData>;

        using FrictionalJointConstraintModel = ::pinocchio::FrictionalJointConstraintModelTpl<Scalar, Options>;
        using FrictionalJointConstraintData = ::pinocchio::FrictionalJointConstraintDataTpl<Scalar, Options>;

        using BilateralPointConstraintModel = ::pinocchio::BilateralPointConstraintModelTpl<Scalar, Options>;
        using BilateralPointConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(BilateralPointConstraintModel);
        using BilateralPointConstraintData = ::pinocchio::BilateralPointConstraintDataTpl<Scalar, Options>;

        using WeldConstraintModel = ::pinocchio::WeldConstraintModelTpl<Scalar, Options>;
        using WeldConstraintData = ::pinocchio::WeldConstraintDataTpl<Scalar, Options>;
        using WeldConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(WeldConstraintModel);

        using JointLimitConstraintModel = ::pinocchio::JointLimitConstraintModelTpl<Scalar, Options>;
        using JointLimitConstraintData = ::pinocchio::JointLimitConstraintDataTpl<Scalar, Options>;
        using BoxSet = ::pinocchio::BoxSetTpl<Scalar>;

        using FrictionalPointConstraintModel = ::pinocchio::FrictionalPointConstraintModelTpl<Scalar, Options>;
        using FrictionalPointConstraintData = ::pinocchio::FrictionalPointConstraintDataTpl<Scalar, Options>;
        using CoulombFrictionCone = ::pinocchio::CoulombFrictionConeTpl<Scalar>;
        using CoulombFrictionConeVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(CoulombFrictionCone);
        using ConstraintCholeskyDecomposition = ::pinocchio::ContactCholeskyDecompositionTpl<Scalar, Options>;
        using ContactPointVector = PINOCCHIO_ALIGNED_STD_VECTOR(Vector3s);
        using ContactNormalVector = PINOCCHIO_ALIGNED_STD_VECTOR(Vector3s);
        using SE3 = ::pinocchio::SE3Tpl<Scalar, Options>;
        using PlacementVector = PINOCCHIO_ALIGNED_STD_VECTOR(SE3);
        // operator to compute contact frames
        using PlacementFromNormalAndPosition = PlacementFromNormalAndPositionTpl<Scalar, Options>;

        /////////////////////////////////////////////////
        /// ATTRIBUTES
        /////////////////////////////////////////////////

        /// ----------------------------------
        /// General attributes

    public:
        /// \brief Cholesky decomposition of the constraints problem.
        ConstraintCholeskyDecomposition constraint_cholesky_decomposition;

        /// \brief Vector of constraint models currently active in the constraints problem.
        WrappedConstraintModelVector constraint_models;

        /// \brief Vector of constraint models' data.
        WrappedConstraintDataVector constraint_datas;

        /// \brief Wether the constraints problem should be treated as a NCP or a CCP.
        /// NCP = Non-linear Complementarity Problem
        /// CCP = Convex Complementarity Problem
        bool is_ncp{true};

    protected:
        /// \brief Handle to the model of the system.
        ModelHandle m_model;

        /// \brief Handle to the model's data of the system.
        DataHandle m_data;

        /// \brief Handle to the geometry model of the system.
        GeometryModelHandle m_geom_model;

        /// \brief Handle to the geometry model's data of the system.
        GeometryDataHandle m_geom_data;

        /// \brief Drift term of the constraints problem.
        VectorXs m_g;

        /// \brief Preconditionner of the constraints problem.
        VectorXs m_preconditioner;

        /// \brief Velocity of constraints.
        VectorXs m_constraints_velocities;

        /// \brief Warm start for the constraint solver
        VectorXs m_constraints_velocities_warmstarts;

        /// \brief Holders for constraint forces (current and previous).
        std::array<VectorXs, 2> m_constraints_forces_holder;

        /// \brief Current constraints problem holder id.
        std::size_t m_current_constraints_pb_id;

        /// \brief Vector of time scaling factors to convert acceleration units to the units of each constraint.
        VectorXs m_time_scaling_acc_to_constraints;

        /// ----------------------------------
        /// Joints dry frictions constraint

    public:
        /// \brief Frictional joint constraint model.
        ConstraintModel joint_friction_constraint_model;

        /// \brief Frictional joint constraint data.
        ConstraintData joint_friction_constraint_data;

        /// ----------------------------------
        /// Bilateral constraints

        /// \brief Holder for bilateral point constraint models.
        ConstraintModelVector bilateral_point_constraint_models;

        /// \brief Holder for bilateral point constraint datas.
        ConstraintDataVector bilateral_point_constraint_datas;

        /// ----------------------------------
        /// Weld constraints

        /// \brief Holder for weld point constraint models.
        ConstraintModelVector weld_constraint_models;

        /// \brief Holder for weld point constraint datas.
        ConstraintDataVector weld_constraint_datas;

        /// ----------------------------------
        /// Joint limits constraint

        /// \brief Joint Limit constraint model.
        ConstraintModel joint_limit_constraint_model;

        /// \brief Joint Limit constraint data.
        ConstraintData joint_limit_constraint_data;

        /// ----------------------------------
        /// Frictional point constraints

        /// \brief Holder for frictional point constraint models.
        ConstraintModelVector frictional_point_constraint_models;

        /// \brief Holder for frictional point constraint datas.
        ConstraintDataVector frictional_point_constraint_datas;

        /// \brief Vector of collision pair ids that are in collision.
        std::vector<std::size_t> pairs_in_collision;

        /// \brief The collision pairs that were in contact at the previous time step (before update function is called).
        std::vector<bool> previous_colliding_collision_pairs;

        /// \brief Vector that maps the id of the contact to the collision pair.
        /// Therefore `contact_id_to_collision_pair[i]` is the id of the collision pair
        /// corresponding to the i-th contact.
        /// Note: since collision pairs can have multiple contacts, the same id can be found
        /// multiple times inside `contact_id_to_collision_pair`.
        std::vector<std::size_t> contact_id_to_collision_pair;

        /// \brief Since each collision pair can have multiple contact points, this struct maps a single
        /// collision pair to contact points information: contact position, contact normal, contact
        /// placements, constraint models/datas, friction cones, elasticities and penetration depths.
        struct ContactMapper
        {
            ContactMapper(std::size_t begin, std::size_t count)
            : begin(begin)
            , count(count)
            {
            }
            /// \brief The id of the first contact point for the considered collision pair.
            std::size_t begin;

            /// \brief Number of contact points for the considered collision pair.
            std::size_t count;
        };

        /// \brief Vector of contact mappers.
        /// The i-th element of this vector is the contact mapper for the i-th colliding collision pair.
        std::vector<ContactMapper> contact_mappers;

        /// \brief A contact can either be:
        /// - breaking (no contact force)
        /// - sticking (contact force inside the friction cone)
        /// - sliding (contact force saturating the friction cone)
        // TODO this should be moved to constraints
        enum struct ContactMode
        {
            BREAKING,
            STICKING,
            SLIDING,
        };

        /// \brief vector of contact modes
        std::vector<ContactMode> contact_modes;

        /// \brief indexes of breaking contacts
        std::vector<ContactIndex> breaking_contacts;

        /// \brief indexes of sticking contacts
        std::vector<ContactIndex> sticking_contacts;

        /// \brief indexes of sliding contacts
        std::vector<ContactIndex> sliding_contacts;

    protected:
        /// \brief Maximum number of contact points per collision pair.
        ContactIndex m_max_num_contact_per_collision_pair;

        /// \brief Number of contact points after `update` is called.
        ContactIndex m_num_contacts;

        /// \brief Number of contact points in previous time step.
        ContactIndex m_previous_num_contacts;

        /// \brief Holders for contact placements (current and previous).
        std::array<PlacementVector, 2> m_frictional_point_constraints_placements_holder;

        /// \brief Size of vector of previous joint limit constraint forces.
        int m_previous_joint_limits_constraint_forces_size;

        /// \brief Active set of previous joint limit constraint model.
        std::vector<std::size_t> m_previous_joint_limit_active_set;

        /////////////////////////////////////////////////
        /// METHODS
        /////////////////////////////////////////////////
    public:
        /// ----------------------------------
        /// Constructors

        /// \brief Default constructor.
        ConstraintsProblemTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle,
            const BilateralPointConstraintModelVector & bilateral_point_constraint_models,
            const WeldConstraintModelVector & weld_constraint_models);

        /// \brief Default constructor.
        ConstraintsProblemTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle,
            const BilateralPointConstraintModelVector & bilateral_point_constraint_models);

        /// \brief Default constructor.
        ConstraintsProblemTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle,
            const WeldConstraintModelVector & weld_constraint_models);

        /// \brief Default constructor.
        ConstraintsProblemTpl(
            const ModelHandle & model_handle,
            const DataHandle & data_handle,
            const GeometryModelHandle & geom_model_handle,
            const GeometryDataHandle & geom_data_handle);

        /// ----------------------------------
        /// General methods

        /// \brief Returns a const reference to the model
        const Model & model() const
        {
            assert(this->m_model != nullptr);
            return pinocchio::helper::get_ref(this->m_model);
        }

        /// \brief Returns a reference to the model
        Model & model()
        {
            assert(this->m_model != nullptr);
            return pinocchio::helper::get_ref(this->m_model);
        }

        /// \brief Returns a const reference to the data
        const Data & data() const
        {
            assert(this->m_data != nullptr);
            return pinocchio::helper::get_ref(this->m_data);
        }

        /// \brief Returns a reference to the data
        Data & data()
        {
            assert(this->m_data != nullptr);
            return pinocchio::helper::get_ref(this->m_data);
        }

        /// \brief Returns a const reference to the geometry model
        const pinocchio::GeometryModel & geom_model() const
        {
            assert(this->m_geom_model != nullptr);
            return pinocchio::helper::get_ref(this->m_geom_model);
        }

        /// \brief Returns a reference to the geometry model
        pinocchio::GeometryModel & geom_model()
        {
            assert(this->m_geom_model != nullptr);
            return pinocchio::helper::get_ref(this->m_geom_model);
        }

        /// \brief Returns a const reference to the geometry data
        const pinocchio::GeometryData & geom_data() const
        {
            assert(this->m_geom_data != nullptr);
            return pinocchio::helper::get_ref(this->m_geom_data);
        }

        /// \brief Returns a reference to the geometry data
        pinocchio::GeometryData & geom_data()
        {
            assert(this->m_geom_data != nullptr);
            return pinocchio::helper::get_ref(this->m_geom_data);
        }

        /// \brief Allocates memory for the constraints problem quantities.
        /// Notes:
        ///   - This method uses the the geometry model's active collision pairs to allocate memory.
        ///   - because we always resize the constraints problem quantities, there won't be any error if
        ///   this method is not called.
        ///     This method is meant to optimize memory allocation for advanced users.
        void allocate();

        /// \brief Empties constraints problem quantities.
        void clear();

        /// \brief After `model`, `data`, `geom_model` and `geom_data` have been updated, this function
        /// updates `constraints`. \param compute_warm_start whether or not to compute a warm-start for
        /// the constraints forces, based on the previous solution of the constraints problem.
        void update(const bool compute_warm_start);

        /// \brief Build the constraints problem quantities: `G`, `g`.
        /// Also builds the quantities necessary to warm-start the constraint solver.
        /// Meant to be called after `update`.
        template<typename FreeVelocityVectorType, typename VelocityVectorType, typename VelocityWarmStartVectorType>
        void build(
            const Eigen::MatrixBase<FreeVelocityVectorType> & vfree,
            const Eigen::MatrixBase<VelocityVectorType> & v,
            const Eigen::MatrixBase<VelocityWarmStartVectorType> & v_warmstart,
            Scalar dt);

        /// \brief Checks consistency of the constraints problem w.r.t to its handles.
        bool check() const;

        /// \brief Size of the constraints problem.
        int constraints_problem_size() const
        {
            return this->joint_friction_constraint_size() //
                   + this->bilateral_constraints_size()   //
                   + this->weld_constraints_size()        //
                   + this->joint_limit_constraint_size()  //
                   + this->frictional_point_constraints_size();
        }

        /// \brief Size of the previous constraints problem.
        int previous_constraints_problem_size() const
        {
            return this->joint_friction_constraint_size()         //
                   + this->bilateral_constraints_size()           //
                   + this->weld_constraints_size()                //
                   + this->previous_joint_limit_constraint_size() //
                   + this->previous_frictional_point_constraints_size();
        }

        /// \brief Getter for the `g` term (i.e. the free velocity of constraints + corrective terms).
        Eigen::VectorBlock<VectorXs> g()
        {
            return this->m_g.head(this->constraints_problem_size());
        }

        /// \brief const getter for the `g` term (i.e. the free velocity of constraint + corrective terms).
        Eigen::VectorBlock<const VectorXs> g() const
        {
            return this->m_g.head(this->constraints_problem_size());
        }

        /// \brief Getter for the preconditioner
        Eigen::VectorBlock<VectorXs> preconditioner()
        {
            return this->m_preconditioner.head(this->constraints_problem_size());
        }

        /// \brief Const getter for the preconditioner
        Eigen::VectorBlock<const VectorXs> preconditioner() const
        {
            return this->m_preconditioner.head(this->constraints_problem_size());
        }

        /// \brief Getter for the time scaling factors to convert acceleration units to the units of each constraint.
        Eigen::VectorBlock<VectorXs> time_scaling_acc_to_constraints()
        {
            return this->m_time_scaling_acc_to_constraints.head(this->constraints_problem_size());
        }

        /// \brief Const getter for the time scaling factors to convert acceleration units to the units of each constraint.
        Eigen::VectorBlock<const VectorXs> time_scaling_acc_to_constraints() const
        {
            return this->m_time_scaling_acc_to_constraints.head(this->constraints_problem_size());
        }

        /// \brief Getter for constraints' forces.
        Eigen::VectorBlock<VectorXs> constraints_forces()
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].head(this->constraints_problem_size());
        }

        /// \brief Const getter for constraints' forces.
        Eigen::VectorBlock<const VectorXs> constraints_forces() const
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].head(this->constraints_problem_size());
        }

        /// \brief Getter for constraints' forces at the previous time step.
        Eigen::VectorBlock<VectorXs> previous_constraints_forces()
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].head(this->previous_constraints_problem_size());
        }

        /// \brief Const getter for constraints' forces at the previous time step.
        Eigen::VectorBlock<const VectorXs> previous_constraints_forces() const
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].head(this->previous_constraints_problem_size());
        }

        /// \brief Getter for constraints' velocities.
        /// note: this should be mutltiplied by dt to retrieve the actual constraints' velocities.
        Eigen::VectorBlock<VectorXs> constraints_velocities()
        {
            return this->m_constraints_velocities.head(this->constraints_problem_size());
        }

        /// \brief Const getter for constraints' velocities.
        /// note: this should be mutltiplied by dt to retrieve the actual constraints' velocities.
        Eigen::VectorBlock<const VectorXs> constraints_velocities() const
        {
            return this->m_constraints_velocities.head(this->constraints_problem_size());
        }

        /// \brief Getter for the the warm-start which uses the constraints' velocities and constraint-inverse dynamics.
        Eigen::VectorBlock<VectorXs> constraints_velocities_warmstarts()
        {
            return this->m_constraints_velocities_warmstarts.head(this->constraints_problem_size());
        }

        /// \brief Const getter for the the warm-start which uses the constraints' velocities and constraint-inverse dynamics.
        Eigen::VectorBlock<const VectorXs> constraints_velocities_warmstarts() const
        {
            return this->m_constraints_velocities_warmstarts.head(this->constraints_problem_size());
        }

    protected:
        /// \brief Updates m_current_constraints_pb_id.
        void updateConstraintsHolders();

        /// \brief Checks consistency of the constraints' model/data vectors w.r.t geom_model and geom_data.
        bool checkConstraintsConsistencyWithGeometryModel() const;

        /// \brief Compute constraint drift g = Jc * vfree + baumgarte.
        /// For each constraint, the drift is expressed in the constraint formulation level
        /// of the constraint (position, velocity or acceleration)
        template<typename FreeVelocityVectorType, typename VelocityVectorType>
        void computeConstraintsDrift(
            const Eigen::MatrixBase<FreeVelocityVectorType> & vfree, const Eigen::MatrixBase<VelocityVectorType> & v, const Scalar dt);

    public:
        /// ----------------------------------
        /// Joints dry frictions constraints

        /// \brief Size of the joints dry friction force vector
        int joint_friction_constraint_size() const
        {
            return this->joint_friction_constraint_model.size();
        }

        /// \brief Return the number of joint friction constraints.
        std::size_t getNumberOfJointFrictionConstraints() const
        {
            if (this->joint_friction_constraint_model.size() > 0)
            {
                return 1;
            }
            return 0;
        }

        /// \brief Getter for friction forces on the joints.
        Eigen::VectorBlock<VectorXs> joint_friction_constraint_forces()
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].head(this->joint_friction_constraint_size());
        }

        /// \brief Const getter for friction forces on the joints.
        Eigen::VectorBlock<const VectorXs> joint_friction_constraint_forces() const
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].head(this->joint_friction_constraint_size());
        }

        /// \brief Getter for previous friction forces on the joints.
        Eigen::VectorBlock<VectorXs> previous_joint_friction_constraint_forces()
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].head(this->joint_friction_constraint_size());
        }

        /// \brief Const getter for previous friction forces on the joints.
        Eigen::VectorBlock<const VectorXs> previous_joint_friction_constraint_forces() const
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].head(this->joint_friction_constraint_size());
        }

        /// ----------------------------------
        /// Bilateral constraints

        /// \brief Size of the bilateral constraints.
        int bilateral_constraints_size() const
        {
            return (int)(3 * this->bilateral_point_constraint_models.size());
        }

        /// \brief Return the number of bilateral constraints.
        std::size_t getNumberOfBilateralConstraints() const
        {
            return this->bilateral_point_constraint_models.size();
        }

        /// \brief Getter for bilateral constraints' forces.
        Eigen::VectorBlock<VectorXs> bilateral_constraints_forces()
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size(), this->bilateral_constraints_size());
        }

        /// \brief Const getter for bilateral constraints' forces.
        Eigen::VectorBlock<const VectorXs> bilateral_constraints_forces() const
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size(), this->bilateral_constraints_size());
        }

        /// \brief Getter for bilateral constraints' forces.
        Eigen::VectorBlock<VectorXs> previous_bilateral_constraints_forces()
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size(), this->bilateral_constraints_size());
        }

        /// \brief Const getter for bilateral constraints' forces.
        Eigen::VectorBlock<const VectorXs> previous_bilateral_constraints_forces() const
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size(), this->bilateral_constraints_size());
        }

        /// ----------------------------------
        /// Weld constraints

        /// \brief Size of the weld constraints.
        int weld_constraints_size() const
        {
            return (int)(6 * this->weld_constraint_models.size());
        }

        /// \brief Return the number of weld constraints.
        std::size_t getNumberOfWeldConstraints() const
        {
            return this->weld_constraint_models.size();
        }

        /// \brief Getter for weld constraints' forces.
        Eigen::VectorBlock<VectorXs> weld_constraints_forces()
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size(), this->weld_constraints_size());
        }

        /// \brief Const getter for weld constraints' forces.
        Eigen::VectorBlock<const VectorXs> weld_constraints_forces() const
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size(), this->weld_constraints_size());
        }

        /// \brief Getter for previous weld constraints' forces.
        Eigen::VectorBlock<VectorXs> previous_weld_constraints_forces()
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size(), this->weld_constraints_size());
        }

        /// \brief Const getter for previous weld constraints' forces.
        Eigen::VectorBlock<const VectorXs> previous_weld_constraints_forces() const
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size(), this->weld_constraints_size());
        }

        /// ----------------------------------
        /// Joint limits constraint

        /// \brief Maximum size of the joints limit force vector.
        int joint_limit_constraint_max_size() const
        {
            return this->joint_limit_constraint_model.size();
        }

        /// \brief Size of the joints limit force vector.
        /// This corresponds to the size of the currently active joint limit constraint (i.e. the distance between the
        /// joint's position and the joint's limits is below a threshold).
        /// This requires the `update` function to be called before (as it itself calls `calc` on the joint limit constraint models/datas).
        int joint_limit_constraint_size() const
        {
            return this->joint_limit_constraint_model.activeSize();
        }

        /// \brief Previous size of the joints limit force vector.
        int previous_joint_limit_constraint_size() const
        {
            return int(this->m_previous_joint_limits_constraint_forces_size);
        }

        /// \brief Return the number of joint limit constraint.
        std::size_t getNumberOfJointLimitConstraints() const
        {
            if (this->joint_limit_constraint_model.activeSize() > 0)
            {
                return 1;
            }
            return 0;
        }

        /// \brief Getter for forces from the limits on the joints.
        Eigen::VectorBlock<VectorXs> joint_limit_constraint_forces()
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size() + this->weld_constraints_size(),
                this->joint_limit_constraint_size());
        }

        /// \brief Const getter for forces from the limits on the joints.
        Eigen::VectorBlock<const VectorXs> joint_limit_constraint_forces() const
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size() + this->weld_constraints_size(),
                this->joint_limit_constraint_size());
        }

        /// \brief Getter for forces from the previous limits on the joints.
        Eigen::VectorBlock<VectorXs> previous_joint_limit_constraint_forces()
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size() + this->weld_constraints_size(),
                this->previous_joint_limit_constraint_size());
        }

        /// \brief Getter for forces from the previous limits on the joints.
        Eigen::VectorBlock<const VectorXs> previous_joint_limit_constraint_forces() const
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size() + this->bilateral_constraints_size() + this->weld_constraints_size(),
                this->previous_joint_limit_constraint_size());
        }

        /// ----------------------------------
        /// Frictional point constraints

        /// \brief Size of vector of frictional point constraints' forces.
        /// For example, if only 3D contacts are considered, this should be 3 * nc, where nc is the
        /// number of contacts obtained after calling the `update` method.
        int frictional_point_constraints_size() const
        {
            return (int)(3 * this->m_num_contacts);
        }

        /// \brief Size of the previous contact problem.
        int previous_frictional_point_constraints_size() const
        {
            return (int)(3 * this->m_previous_num_contacts);
        }

        /// \brief Sets the maximum number of contact points per collision pair.
        void setMaxNumberOfContactsPerCollisionPair(ContactIndex num)
        {
            if (num <= 0)
            {
                PINOCCHIO_THROW_PRETTY(std::logic_error, "Cannot set max number of contacts per collision pair to a value <= 0.");
            }
            this->m_max_num_contact_per_collision_pair = num;
            this->allocate();
        }

        /// \brief Sets the maximum number of contact points per collision pair.
        ContactIndex getMaxNumberOfContactsPerCollisionPair() const
        {
            return this->m_max_num_contact_per_collision_pair;
        }

        /// \brief Returns the maximum number of contact points this `ConstraintsProblem` can handle.
        /// This number determines the allocated memory of the `ConstraintsProblem`.
        /// To update the maximum number of contact points, please update the geom_model and call the
        /// `allocate` method to compute the updated maximum number of contact points.
        ContactIndex getMaxNumberOfContacts() const
        {
            return this->frictional_point_constraint_models.size();
        }

        /// \brief Returns the number of contact points after `update` is called.
        ContactIndex getNumberOfContacts() const
        {
            return this->m_num_contacts;
        }

        /// \brief Returns the previous number of contact points after `update` is called.
        ContactIndex getPreviousNumberOfContacts() const
        {
            return this->m_previous_num_contacts;
        }

        /// \brief Getter for frictional point constraints' forces.
        Eigen::VectorBlock<VectorXs> frictional_point_constraints_forces()
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Const getter for frictional point constraints' forces.
        Eigen::VectorBlock<const VectorXs> frictional_point_constraints_forces() const
        {
            return this->m_constraints_forces_holder[this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Getter for frictional point constraints' forces at the previous time step.
        Eigen::VectorBlock<VectorXs> previous_frictional_point_constraints_forces()
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->previous_joint_limit_constraint_size(),
                this->previous_frictional_point_constraints_size());
        }

        /// \brief Const getter for frictional point constraints' forces at the previous time step.
        Eigen::VectorBlock<const VectorXs> previous_frictional_point_constraints_forces() const
        {
            return this->m_constraints_forces_holder[1 - this->m_current_constraints_pb_id].segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->previous_joint_limit_constraint_size(),
                this->previous_frictional_point_constraints_size());
        }

        /// \brief Getter for frictional point constraints' velocities.
        /// note: this should be mutltiplied by dt to retrieve the actual velocities.
        Eigen::VectorBlock<VectorXs> frictional_point_constraints_velocities()
        {
            return this->m_constraints_velocities.segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Const getter for frictional point constraints' velocities.
        /// note: this should be mutltiplied by dt to retrieve the actual velocities.
        Eigen::VectorBlock<const VectorXs> frictional_point_constraints_velocities() const
        {
            return this->m_constraints_velocities.segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Getter for the the warm-start which uses the frictional point constraints' velocities and contact-inverse dynamics.
        Eigen::VectorBlock<VectorXs> frictional_point_constraints_warmstart()
        {
            return this->m_constraints_velocities_warmstarts.segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Const getter for the the warm-start which uses the frictional point constraints' velocities and contact-inverse dynamics.
        Eigen::VectorBlock<const VectorXs> frictional_point_constraints_warmstart() const
        {
            return this->m_constraints_velocities_warmstarts.segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Getter for the time scaling factors to convert acceleration units to the units of each constraint.
        Eigen::VectorBlock<VectorXs> contact_time_scaling_acc_to_constraints()
        {
            return this->m_time_scaling_acc_to_constraints.segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Const getter for the time scaling factors to convert acceleration units to the units of each constraint.
        Eigen::VectorBlock<const VectorXs> contact_time_scaling_acc_to_constraints() const
        {
            return this->m_time_scaling_acc_to_constraints.segment(
                this->joint_friction_constraint_size()   //
                    + this->bilateral_constraints_size() //
                    + this->weld_constraints_size()      //
                    + this->joint_limit_constraint_size(),
                this->frictional_point_constraints_size());
        }

        /// \brief Getter for contact placements.
        PlacementVector & frictional_point_constraints_placements()
        {
            return this->m_frictional_point_constraints_placements_holder[this->m_current_constraints_pb_id];
        }

        /// \brief Const getter for contact placements.
        const PlacementVector & frictional_point_constraints_placements() const
        {
            return this->m_frictional_point_constraints_placements_holder[this->m_current_constraints_pb_id];
        }

        /// \brief Getter for contact placements at the previous time step.
        PlacementVector & previous_frictional_point_constraints_placements()
        {
            return this->m_frictional_point_constraints_placements_holder[1 - this->m_current_constraints_pb_id];
        }

        /// \brief Const getter for contact placements at the previous time step.
        const PlacementVector & previous_frictional_point_constraints_placements() const
        {
            return this->m_frictional_point_constraints_placements_holder[1 - this->m_current_constraints_pb_id];
        }

        /// \brief Collecting active set from the solution of the contact problem.
        /// the contact problem should be solved before calling this method.
        void collectActiveSet(Scalar epsilon = 1e-6);
    };

} // namespace simplex

/* --- Details -------------------------------------------------------------- */
#include "simplex/core/constraints-problem.hxx"

#if SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION
    #include "simplex/core/constraints-problem.txx"
#endif

#endif // ifndef __simplex_core_constraints_problem_hpp__
