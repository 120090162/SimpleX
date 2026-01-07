#ifndef __simplex_core_ncp_derivatives_hpp__
#define __simplex_core_ncp_derivatives_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/macros.hpp"
#include "simplex/core/constraints-problem-derivatives.hpp"

namespace simplex
{

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct traits<ContactSolverDerivativesTpl<_Scalar, _Options, JointCollectionTpl>>
    {
        using Scalar = _Scalar;
    };

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct ContactSolverDerivativesTpl
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // -------------------------------------------------------------------------------------------------
        // TYPEDEFS
        // -------------------------------------------------------------------------------------------------
        using ConstraintsProblemDerivatives = ConstraintsProblemDerivativesTpl<_Scalar, _Options, JointCollectionTpl>;
        using Scalar = _Scalar;
        enum
        {
            Options = _Options
        };

        using VectorXs = typename ConstraintsProblemDerivatives::VectorXs;
        using Vector3s = typename ConstraintsProblemDerivatives::Vector3s;
        using Vector2s = Eigen::Matrix<Scalar, 2, 1, Options>;
        using Matrix23s = Eigen::Matrix<Scalar, 2, 3, Options>;
        using Matrix32s = Eigen::Matrix<Scalar, 3, 2, Options>;
        using MatrixXs = typename ConstraintsProblemDerivatives::MatrixXs;

        using ConstraintsProblemDerivativesHandle = std::shared_ptr<ConstraintsProblemDerivatives>;
        using ContactIndex = typename ConstraintsProblemDerivatives::ContactIndex;
        using ContactIndexVector = std::vector<ContactIndex>;

        using FrictionalPointConstraintModel = typename ConstraintsProblemDerivatives::FrictionalPointConstraintModel;
        using ConstraintModel = typename ConstraintsProblemDerivatives::ConstraintModel;

        SIMPLEX_PUBLIC
        /// \brief Whether or not timings of the `step` function are measured.
        /// If set to true, the timing of the last call to `step` can be accessed via `getCPUTimes`
        bool measure_timings{false};

        /// \brief Which solver to use to solve the system of implicit gradients.
        enum struct ImplicitGradientSystemSolver
        {
            HOUSEHOLDER_QR, // Faster
            COD,            // More stable
        } implicit_gradient_solver_type;

        SIMPLEX_PROTECTED
        /// \brief Contact problem to differentiate.
        ConstraintsProblemDerivativesHandle constraint_problem_;

        /// \brief Maximum number of dimension of parameters (theta)
        int max_theta_dim_;

        /// \brief Matrix to inverse for implicit gradients.
        /// \note reordered delassus to match the active set of the contact problem.
        VectorXs G_data_;

        /// \brief right hand side of the system to solve for implicit gradients.
        VectorXs rhs_data_;

        /// \brief Derivatives of lambda i.e. the solution of the NCP, expressed in minimal coordinates
        /// (in the planes tangent to the friction cones for sliding contact points).
        VectorXs dlam_dtheta_reduced_data_;

        /// \brief Derivatives of lambda i.e. the solution of the NCP
        VectorXs dlam_dtheta_data_;

        /// \brief Matrix containing the delassus matrix of the contact problem
        VectorXs delassus_data_;

        /// \brief Matrix containing the basis of 2D planes that are tangent to the friction cone for
        /// sliding contacts.
        PINOCCHIO_ALIGNED_STD_VECTOR(Matrix32s) E_;

        /// \brief Matrix containing the basis of 2D planes that are tangent to the friction cone for
        /// sliding contacts.
        PINOCCHIO_ALIGNED_STD_VECTOR(Matrix23s) ETP_;

        /// \brief size of the problem on implicit gradients
        Eigen::Index gradients_problem_size_;

        /// \brief size of the parameter vector (theta)
        Eigen::Index theta_size_;

        /// \brief Timer used for evaluating timings of jvp.
        hpp::fcl::Timer timer_;

        /// \brief Timings for the call to the contact solver.
        hpp::fcl::CPUTimes contact_solver_derivatives_timings_;

        /// \brief COD solver for the system of implicit gradients.
        std::shared_ptr<Eigen::CompleteOrthogonalDecomposition<Eigen::Ref<MatrixXs>>> cod_solver_;

        /// \brief QR solver for the system of implicit gradients.
        std::shared_ptr<Eigen::HouseholderQR<Eigen::Ref<MatrixXs>>> qr_solver_;

        SIMPLEX_PUBLIC
        ///
        /// \brief Default constructor
        ContactSolverDerivativesTpl(ConstraintsProblemDerivativesHandle constraint_problem)
        : measure_timings(false)
        , implicit_gradient_solver_type(ImplicitGradientSystemSolver::COD) // COD by default, more stable
        , constraint_problem_(constraint_problem)
        , timer_(false)
        , cod_solver_(nullptr)
        , qr_solver_(nullptr)
        {
            max_theta_dim_ = 3 * constraint_problem->model().nv;
            allocate();
        }

        ///
        /// \brief Compute the system of implicit equations
        void compute();

        ///
        /// \brief Computes Jacobian vector product of the solution of the NCP associated to the contact
        /// problem.
        /// \note This method should be called after the contact problem has been solved.
        template<typename Matrix>
        void jvp(const Eigen::MatrixBase<Matrix> & dGlamgdtheta);

        ///
        /// \brief Getter for the right hand side of the system of implicit gradients
        Eigen::Map<MatrixXs> rhs()
        {
            return Eigen::Map<MatrixXs>(rhs_data_.data(), gradients_problem_size_, theta_size_);
        }

        ///
        /// \brief Const getter for the right hand side of the system of implicit gradients
        const Eigen::Map<const MatrixXs> rhs() const
        {
            return Eigen::Map<const MatrixXs>(rhs_data_.data(), gradients_problem_size_, theta_size_);
        }

        ///
        /// \brief Getter for the matrix of the system of implicit gradients
        Eigen::Map<MatrixXs> G()
        {
            return Eigen::Map<MatrixXs>(G_data_.data(), gradients_problem_size_, gradients_problem_size_);
        }

        ///
        /// \brief Const getter for the matrix of the system of implicit gradients
        const Eigen::Map<const MatrixXs> G() const
        {
            return Eigen::Map<const MatrixXs>(G_data_.data(), gradients_problem_size_, gradients_problem_size_);
        }

        ///
        /// \brief Getter for the reduced gradients dlambdadtheta_reduced
        Eigen::Map<MatrixXs> dlam_dtheta_reduced()
        {
            return Eigen::Map<MatrixXs>(dlam_dtheta_reduced_data_.data(), gradients_problem_size_, theta_size_);
        }

        ///
        /// \brief Const getter for the reduced gradients dlambdadtheta_reduced
        const Eigen::Map<const MatrixXs> dlam_dtheta_reduced() const
        {
            return Eigen::Map<const MatrixXs>(dlam_dtheta_reduced_data_.data(), gradients_problem_size_, theta_size_);
        }

        ///
        /// \brief Getter for the gradients dlambdadtheta
        Eigen::Map<MatrixXs> dlam_dtheta()
        {
            const Eigen::Index idx = (Eigen::Index)(3 * constraint_problem_->getNumberOfContacts()); // rows
            return Eigen::Map<MatrixXs>(dlam_dtheta_data_.data(), idx, theta_size_);
        }

        ///
        /// \brief Const getter for the gradients dlambdadtheta
        const Eigen::Map<const MatrixXs> dlam_dtheta() const
        {
            const Eigen::Index idx = (Eigen::Index)(3 * constraint_problem_->getNumberOfContacts());
            return Eigen::Map<const MatrixXs>(dlam_dtheta_data_.data(), idx, theta_size_);
        }

        /// \brief Get timings of the call to the contact solver derivatives.
        /// This timing is set to 0 if there was no contacts.
        hpp::fcl::CPUTimes getContactSolverDerivativesCPUTimes() const
        {
            return contact_solver_derivatives_timings_;
        }

        ///
        /// \brief Returns a const reference to the contact problem
        const ConstraintsProblemDerivatives & constraint_problem() const
        {
            return pinocchio::helper::get_ref(constraint_problem_);
        }

        ///
        /// \brief Returns a const reference to the contact problem
        ConstraintsProblemDerivatives & constraint_problem()
        {
            return pinocchio::helper::get_ref(constraint_problem_);
        }

        SIMPLEX_PROTECTED
        ///
        /// \brief Allocates memory based on `model` and active collision pairs in `geom_model`.
        void allocate();

        ///
        /// \brief Getter for the delassus matrix
        Eigen::Map<MatrixXs> delassus()
        {
            const Eigen::Index idx = (Eigen::Index)(3 * constraint_problem_->getNumberOfContacts());
            return Eigen::Map<MatrixXs>(delassus_data_.data(), idx, idx);
        }

        ///
        /// \brief Const getter for the delassus matrix
        const Eigen::Map<const MatrixXs> delassus() const
        {
            const Eigen::Index idx = (Eigen::Index)(3 * constraint_problem_->getNumberOfContacts());
            return Eigen::Map<const MatrixXs>(delassus_data_.data(), idx, idx);
        }

        ///
        /// \brief Getter for the matrix E of the basis of 2D planes that are tangent to the friction
        /// cone for sliding contacts.
        PINOCCHIO_ALIGNED_STD_VECTOR(Matrix32s) & E()
        {
            return E_;
        }

        ///
        /// \brief Const getter for the matrix E of the basis of 2D planes that are tangent to the
        /// friction cone for sliding contacts.
        const PINOCCHIO_ALIGNED_STD_VECTOR(Matrix32s) & E() const
        {
            return E_;
        }

        ///
        /// \brief Getter for the matrix ETP of the basis of 2D planes that are tangent to the friction
        /// cone for sliding contacts.
        PINOCCHIO_ALIGNED_STD_VECTOR(Matrix23s) & ETP()
        {
            return ETP_;
        }

        ///
        /// \brief Const getter for the matrix ETP of the basis of 2D planes that are tangent to the
        /// friction cone for sliding contacts.
        const PINOCCHIO_ALIGNED_STD_VECTOR(Matrix23s) & ETP() const
        {
            return ETP_;
        }

        ///
        /// \brief Build system of implicit gradients.
        void buildSystemOfImplicitGrads();

        template<typename MatrixIn, typename MatrixOut>
        void expressInMinimalSystem(const Eigen::MatrixBase<MatrixIn> & rhs, const Eigen::MatrixBase<MatrixOut> & reduced_rhs) const;

        template<typename MatrixIn, typename MatrixOut>
        void extractFromMinimalSystem(
            const Eigen::MatrixBase<MatrixIn> & reduced_sol, const Eigen::MatrixBase<MatrixOut> & dlambda_dtheta) const;
    };

} // namespace simplex

/* --- Details -------------------------------------------------------------- */
#include "simplex/core/ncp-derivatives.hxx"

#if SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION
    #include "simplex/core/ncp-derivatives.txx"
#endif

#endif // ifndef __simplex_core_ncp_derivatives_hpp__
