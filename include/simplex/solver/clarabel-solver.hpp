#ifndef __simplex_solver_clarabel_solver_hpp__
#define __simplex_solver_clarabel_solver_hpp__

#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT

    #include "simplex/core/fwd.hpp"
    #include "simplex/macros.hpp"

    #include <pinocchio/algorithm/constraints/constraint-model-generic.hpp>
    #include <pinocchio/algorithm/constraints/constraint-data-generic.hpp>
    #include <pinocchio/algorithm/delassus-operator-base.hpp>

    #include <Eigen/Core>
    #include <Eigen/Sparse>
    #include <clarabel.hpp>
    #include <boost/optional.hpp>

    #include <vector>
    #include <memory>

namespace simplex
{

    ///
    /// \brief Clarabel-based cone solver wrapper for contact constraint resolution.
    ///
    /// This class provides a Pinocchio-compatible interface to the Clarabel interior point solver
    /// for solving contact problems with second-order cone constraints (friction cones).
    ///
    template<typename _Scalar, int _Options = 0>
    struct ClarabelContactSolverTpl
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        using Scalar = _Scalar;
        enum
        {
            Options = _Options
        };
        using VectorXs = Eigen::Matrix<Scalar, Eigen::Dynamic, 1, Options>;
        using MatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options>;
        using SparseMatrix = Eigen::SparseMatrix<Scalar, Eigen::ColMajor>;
        using Vector3s = Eigen::Matrix<Scalar, 3, 1, Options>;
        using Matrix3s = Eigen::Matrix<Scalar, 3, 3, Options>;

        /// \brief Size of the constraint problem
        int problem_size_;

        /// \brief Primal solution (constraint impulses)
        VectorXs primal_solution_;
        VectorXs prev_x_;
        VectorXs prev_z_;

        /// \brief De Saxcé shift
        VectorXs shift_;
        VectorXs prev_shift_;

        /// \brief Dual solution (constraint velocities)
        VectorXs dual_solution_;

        /// \brief Solver initialized flag
        bool is_initialized_;

        /// \brief Number of iterations from last solve
        int num_iterations_;

        /// \brief Clarabel solver instance (shared_ptr to allow copying)
        std::shared_ptr<clarabel::DefaultSolver<Scalar>> clarabel_solver_;

        /// \brief Default constructor
        ClarabelContactSolverTpl()
        : problem_size_(0)
        , is_initialized_(false)
        , num_iterations_(0)
        {
        }

        /// \brief Constructor with problem size
        explicit ClarabelContactSolverTpl(int problem_size)
        : problem_size_(problem_size)
        , primal_solution_(VectorXs::Zero(problem_size))
        , prev_x_(VectorXs::Zero(problem_size))
        , prev_z_(VectorXs::Zero(problem_size))
        , shift_(VectorXs::Zero(problem_size))
        , prev_shift_(VectorXs::Zero(problem_size))
        , dual_solution_(VectorXs::Zero(problem_size))
        , is_initialized_(false)
        , num_iterations_(0)
        {
            assert(problem_size >= 0);
        }

        /// \brief Set maximum number of iterations
        void setMaxIterations(int max_iter)
        {
            // Will be used when creating solver settings
            max_iterations_ = max_iter;
        }

        /// \brief Set absolute precision tolerance
        void setAbsolutePrecision(Scalar tol)
        {
            absolute_tolerance_ = tol;
        }

        /// \brief Set relative precision tolerance
        void setRelativePrecision(Scalar tol)
        {
            relative_tolerance_ = tol;
        }

        int getIterationCount() const
        {
            return num_iterations_;
        }

        /// \brief Get primal solution (constraint impulses)
        const VectorXs & getPrimalSolution() const
        {
            return primal_solution_;
        }

        /// \brief Get dual solution (constraint velocities)
        const VectorXs & getDualSolution() const
        {
            return dual_solution_;
        }

        /// \brief Check if solver is initialized
        bool isInitialized() const
        {
            return is_initialized_;
        }

        /// \brief Reset the solver
        void reset()
        {
            is_initialized_ = false;
            num_iterations_ = 0;
            primal_solution_.setZero();
            dual_solution_.setZero();
        }

        ///
        /// \brief Solve the contact problem using Clarabel
        ///
        /// The problem is formulated as:
        ///   minimize    (1/2) lambda^T G lambda + g^T lambda
        ///   subject to  lambda in K
        ///
        /// where K is a product of second-order cones (friction cones) and other constraint sets.
        ///
        /// \param[in] delassus The Delassus operator G
        /// \param[in] g The drift term
        /// \param[in] constraint_models Vector of constraint models
        /// \param[in] constraint_datas Vector of constraint data
        /// \param[in] dt Time step
        /// \param[in] preconditioner Optional preconditioner
        /// \param[in] primal_guess Optional initial guess for primal variable
        /// \param[in] dual_guess Optional initial guess for dual variable
        /// \param[in] is_ncp Whether this is a nonlinear complementarity problem
        /// \param[in] record_stats Whether to record solver statistics
        ///
        template<typename DelassusOperator, typename VectorLike, typename ConstraintModelAllocator, typename ConstraintDataAllocator>
        void solve(
            const DelassusOperator & delassus,
            const Eigen::MatrixBase<VectorLike> & g,
            const std::vector<std::reference_wrapper<const ::pinocchio::ConstraintModelTpl<Scalar, Options>>, ConstraintModelAllocator> &
                constraint_models,
            const std::vector<std::reference_wrapper<const ::pinocchio::ConstraintDataTpl<Scalar, Options>>, ConstraintDataAllocator> &
                constraint_datas,
            const boost::optional<Eigen::Ref<const VectorXs>> & preconditioner = boost::none,
            const boost::optional<Eigen::Ref<const VectorXs>> & primal_guess = boost::none,
            const boost::optional<Eigen::Ref<const VectorXs>> & dual_guess = boost::none,
            bool is_ncp = true,
            bool record_stats = false);

    protected:
        int max_iterations_{1000};
        Scalar absolute_tolerance_{1e-8};
        Scalar relative_tolerance_{1e-8};
    };

} // namespace simplex

    // Include implementation
    #include "simplex/solver/clarabel-solver.hxx"

#endif

#endif // __simplex_solver_clarabel_solver_hpp__
