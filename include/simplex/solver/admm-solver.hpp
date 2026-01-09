#ifndef __simplex_solver_admm_solver_hpp__
#define __simplex_solver_admm_solver_hpp__

#include "pinocchio/algorithm/constraints/fwd.hpp"
#include "pinocchio/math/fwd.hpp"
#include "pinocchio/math/comparison-operators.hpp"
#include "pinocchio/math/eigenvalues.hpp"

#include "pinocchio/algorithm/contact-solver-base.hpp"
#include "pinocchio/algorithm/delassus-operator-base.hpp"

#include "pinocchio/math/lanczos-decomposition.hpp"

#include "pinocchio/algorithm/diagonal-preconditioner.hpp"

#include <limits>
#include <boost/optional.hpp>

namespace simplex
{
    template<typename _Scalar>
    struct ADMMContactSolverTpl;

    template<typename _Scalar>
    struct ADMMSpectralUpdateRuleTpl
    {
        using Scalar = _Scalar;

        ADMMSpectralUpdateRuleTpl(
            const Scalar ratio_primal_dual,
            const Scalar L,
            const Scalar m,
            const Scalar rho_power_factor,
            const Scalar rho_update_ratio = Scalar(0))
        : ratio_primal_dual(ratio_primal_dual)
        , rho_update_ratio(rho_update_ratio)
        , rho_increment(std::pow(L / m, rho_power_factor))
        {
            PINOCCHIO_CHECK_INPUT_ARGUMENT(m > Scalar(0), "m should be positive.");
        }

        Scalar getRatioPrimalDual() const
        {
            return ratio_primal_dual;
        }
        void setRatioPrimalDual(const Scalar ratio_primal_dual)
        {
            this->ratio_primal_dual = ratio_primal_dual;
        }

        Scalar getRhoIncrement() const
        {
            return rho_increment;
        }
        void setRhoIncrement(const Scalar cond, const Scalar rho_power_factor)
        {
            rho_increment = std::pow(cond, rho_power_factor);
        }

        bool eval(const Scalar primal_feasibility, const Scalar dual_feasibility, Scalar & rho) const
        {
            bool rho_has_changed = false;
            const Scalar & R = ratio_primal_dual;
            const Scalar & mu = rho_update_ratio > 0 ? rho_update_ratio : ratio_primal_dual;

            const Scalar primal_limit = (rho_update_ratio > 0) ? (R * mu) : mu;
            const Scalar dual_limit = (rho_update_ratio > 0) ? (mu / R) : mu;

            if (primal_feasibility > primal_limit * dual_feasibility)
            {
                rho *= rho_increment;
                //        rho *= math::pow(cond,rho_power_factor);
                //        rho_power += rho_power_factor;
                rho_has_changed = true;
            }
            else if (dual_feasibility > dual_limit * primal_feasibility)
            {
                rho /= rho_increment;
                //        rho *= math::pow(cond,-rho_power_factor);
                //        rho_power -= rho_power_factor;
                rho_has_changed = true;
            }

            return rho_has_changed;
        }

        /// \brief Compute the penalty ADMM value from the current largest and lowest eigenvalues and
        /// the scaling spectral factor.
        static inline Scalar computeRho(const Scalar L, const Scalar m, const Scalar rho_power)
        {
            const Scalar cond = L / m;
            const Scalar rho = pinocchio::math::sqrt(L * m) * pinocchio::math::pow(cond, rho_power);
            return rho;
        }

        /// \brief Compute the  scaling spectral factor of the ADMM penalty term from the current
        /// largest and lowest eigenvalues and the ADMM penalty term.
        static inline Scalar computeRhoPower(const Scalar L, const Scalar m, const Scalar rho)
        {
            const Scalar cond = L / m;
            const Scalar sqrt_L_m = pinocchio::math::sqrt(L * m);
            const Scalar rho_power = pinocchio::math::log(rho / sqrt_L_m) / pinocchio::math::log(cond);
            return rho_power;
        }

    protected:
        Scalar ratio_primal_dual;
        Scalar rho_update_ratio;
        Scalar rho_increment;
    };

    template<typename _Scalar>
    struct ADMMLinearUpdateRuleTpl
    {
        using Scalar = _Scalar;

        ADMMLinearUpdateRuleTpl(
            const Scalar ratio_primal_dual, const Scalar increase_factor, const Scalar decrease_factor, const Scalar rho_update_ratio)
        : ratio_primal_dual(ratio_primal_dual)
        , rho_update_ratio(rho_update_ratio)
        , increase_factor(increase_factor)
        , decrease_factor(decrease_factor)
        {
            PINOCCHIO_CHECK_INPUT_ARGUMENT(increase_factor > Scalar(1), "increase_factor should be greater than one.");
            PINOCCHIO_CHECK_INPUT_ARGUMENT(decrease_factor > Scalar(1), "decrease_factor should be greater than one.");
        }

        ADMMLinearUpdateRuleTpl(const Scalar ratio_primal_dual, const Scalar factor, const Scalar rho_update_ratio)
        : ratio_primal_dual(ratio_primal_dual)
        , rho_update_ratio(rho_update_ratio)
        , increase_factor(factor)
        , decrease_factor(factor)
        {
            PINOCCHIO_CHECK_INPUT_ARGUMENT(factor > Scalar(1), "factor should be greater than one.");
        }

        bool eval(const Scalar primal_feasibility, const Scalar dual_feasibility, Scalar & rho) const
        {
            bool rho_has_changed = false;
            const Scalar & R = ratio_primal_dual;
            const Scalar & mu = rho_update_ratio > 0 ? rho_update_ratio : ratio_primal_dual;

            const Scalar primal_limit = (rho_update_ratio > 0) ? (R * mu) : mu;
            const Scalar dual_limit = (rho_update_ratio > 0) ? (mu / R) : mu;

            if (primal_feasibility > primal_limit * dual_feasibility)
            {
                rho *= increase_factor;
                rho_has_changed = true;
            }
            else if (dual_feasibility > dual_limit * primal_feasibility)
            {
                rho /= decrease_factor;
                rho_has_changed = true;
            }

            return rho_has_changed;
        }

    protected:
        Scalar ratio_primal_dual;
        Scalar rho_update_ratio;
        Scalar increase_factor, decrease_factor;
    };

    enum class ADMMUpdateRule : char
    {
        SPECTRAL = 'S',
        LINEAR = 'L',
        CONSTANT = 'C',
    };

    template<typename Scalar>
    union ADMMUpdateRuleContainerTpl {
        ADMMUpdateRuleContainerTpl()
        : dummy() {};
        ADMMSpectralUpdateRuleTpl<Scalar> spectral_rule;
        ADMMLinearUpdateRuleTpl<Scalar> linear_rule;

    protected:
        struct Dummy
        {
            Dummy() {};
        };

        Dummy dummy{};
    };

    template<typename _Scalar>
    struct PINOCCHIO_UNSUPPORTED_MESSAGE("The API will change towards more flexibility") ADMMContactSolverTpl
    : pinocchio::ContactSolverBaseTpl<_Scalar>
    {
        using Scalar = _Scalar;
        typedef pinocchio::ContactSolverBaseTpl<Scalar> Base;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
        typedef Eigen::Ref<VectorXs> RefVectorXs;
        typedef Eigen::Ref<const VectorXs> RefConstVectorXs;
        typedef const Eigen::Ref<const VectorXs> ConstRefVectorXs;
        typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
        typedef pinocchio::LanczosDecompositionTpl<MatrixXs> LanczosDecomposition;
        typedef pinocchio::DiagonalPreconditionerTpl<VectorXs> DiagonalPreconditioner;

        using Base::problem_size;

        //    struct SolverParameters
        //    {
        //      explicit SolverParameters(const int problem_dim)
        //      : rho_power(Scalar(0.2))
        //      , ratio_primal_dual(Scalar(10))
        //      , mu_prox
        //      {
        //
        //      }
        //
        //      /// \brief Rho solver ADMM
        //      boost::optional<Scalar> rho;
        //      /// \brief Power value associated to rho. This quantity will be automatically updated.
        //      Scalar rho_power;
        //      /// \brief Ratio primal/dual
        //      Scalar ratio_primal_dual;
        //      /// \brief Proximal value
        //      Scalar mu_prox;
        //
        //      /// \brief Largest eigenvalue
        //      boost::optional<Scalar> L_value;
        //      /// \brief Largest eigenvector
        //      boost::optional<VectorXs> L_vector;
        //    };
        //
        struct ADMMSolverStats : Base::SolverStats
        {
            ADMMSolverStats()
            : Base::SolverStats()
            , cholesky_update_count(0)
            {
            }

            explicit ADMMSolverStats(const int max_it)
            : Base::SolverStats(max_it)
            , cholesky_update_count(0)
            {
                reserve(max_it);
            }

            void reserve(const int max_it)
            {
                dual_feasibility_admm.reserve(size_t(max_it));
                dual_feasibility_constraint.reserve(size_t(max_it));
                rho.reserve(size_t(max_it));
            }

            void reset()
            {
                Base::SolverStats::reset();
                dual_feasibility_admm.clear();
                dual_feasibility_constraint.clear();
                rho.clear();
                cholesky_update_count = 0;
            }

            ///  \brief Number of Cholesky updates.
            int cholesky_update_count;

            /// \brief ADMM dual feasibility
            std::vector<Scalar> dual_feasibility_admm;
            /// \brief ADMM dual feasibility
            std::vector<Scalar> dual_feasibility_constraint;

            /// \brief History of rho values.
            std::vector<Scalar> rho;
        };

        //
        //    struct SolverResults
        //    {
        //      explicit SolverResults(const int problem_dim, const int max_it)
        //      : L_vector(problem_dim)
        //
        //      /// \brief Largest eigenvalue
        //      Scalar L_value;
        //      /// \brief Largest eigenvector
        //      VectorXs L_vector;
        //
        //      SolverStats stats;
        //    };
        explicit ADMMContactSolverTpl(
            int problem_dim,
            Scalar mu_prox = Scalar(1e-6),
            Scalar tau = Scalar(0.5),
            Scalar rho_power = Scalar(0.2),
            Scalar rho_power_factor = Scalar(0.05),
            Scalar linear_update_rule_factor = Scalar(2),
            Scalar ratio_primal_dual = Scalar(10),
            int lanczos_size = int(3),
            int max_delassus_decomposition_updates = std::numeric_limits<int>::infinity(),
            Scalar dual_momentum = Scalar(0),
            Scalar rho_momentum = Scalar(0),
            Scalar rho_update_ratio = Scalar(0),
            int rho_min_update_frequency = 1)
        : Base(problem_dim)
        , is_initialized(false)
        , mu_prox(mu_prox)
        , tau(tau)
        , rho(10.)
        , rho_power(rho_power)
        , rho_power_factor(rho_power_factor)
        , linear_update_rule_factor(linear_update_rule_factor)
        , ratio_primal_dual(ratio_primal_dual)
        , max_delassus_decomposition_updates(max_delassus_decomposition_updates)
        , dual_momentum(dual_momentum)
        , rho_momentum(rho_momentum)
        , rho_update_ratio(rho_update_ratio)
        , rho_min_update_frequency(rho_min_update_frequency)
        , lanczos_decomposition(
              static_cast<Eigen::DenseIndex>(pinocchio::math::max(2, problem_dim)),
              static_cast<Eigen::DenseIndex>(pinocchio::math::max(2, pinocchio::math::min(lanczos_size, problem_dim))))
        , x_(VectorXs::Zero(problem_dim))
        , y_(VectorXs::Zero(problem_dim))
        , x_bar_(VectorXs::Zero(problem_dim))
        , y_bar_(VectorXs::Zero(problem_dim))
        , x_bar_previous(VectorXs::Zero(problem_dim))
        , y_bar_previous(VectorXs::Zero(problem_dim))
        , z_bar_previous(VectorXs::Zero(problem_dim))
        , z_(VectorXs::Zero(problem_dim))
        , z_constraint_(VectorXs::Zero(problem_dim))
        , z_bar_(VectorXs::Zero(problem_dim))
        , s_(VectorXs::Zero(problem_dim))
        , s_constraint_(VectorXs::Zero(problem_dim))
        , s_bar_(VectorXs::Zero(problem_dim))
        , preconditioner_(VectorXs::Ones(problem_dim))
        , g_bar_(VectorXs::Zero(problem_dim))
        , time_scaling_acc_to_constraints(VectorXs::Zero(problem_dim))
        , time_scaling_constraints_to_pos(VectorXs::Zero(problem_dim))
        , gs(VectorXs::Zero(problem_dim))
        , rhs(problem_dim)
        , primal_feasibility_vector(VectorXs::Zero(problem_dim))
        , primal_feasibility_vector_bar(VectorXs::Zero(problem_dim))
        , dual_feasibility_vector(VectorXs::Zero(problem_dim))
        , dual_feasibility_vector_bar(VectorXs::Zero(problem_dim))
        , stats()
        {
        }

        /// \brief Set the ADMM penalty value.
        void setRho(const Scalar rho)
        {
            this->rho = rho;
        }
        /// \brief Get the ADMM penalty value.
        Scalar getRho() const
        {
            return rho;
        }

        /// \brief Set the power associated to the problem conditionning.
        void setRhoPower(const Scalar rho_power)
        {
            this->rho_power = rho_power;
        }
        /// \brief Get the power associated to the problem conditionning.
        Scalar getRhoPower() const
        {
            return rho_power;
        }

        /// \brief Set the power factor associated to the problem conditionning.
        void setRhoPowerFactor(const Scalar rho_power_factor)
        {
            this->rho_power_factor = rho_power_factor;
        }
        /// \brief Get the value of the increase/decrease factor associated to the problem
        /// conditionning.
        Scalar getRhoPowerFactor() const
        {
            return rho_power_factor;
        }

        /// \brief Set the update factor of the Linear update rule
        void setLinearUpdateRuleFactor(const Scalar linear_update_rule_factor)
        {
            this->linear_update_rule_factor = linear_update_rule_factor;
        }
        /// \brief Get the value of the increase/decrease factor of the Linear update rule
        Scalar getLinearUpdateRuleFactor() const
        {
            return linear_update_rule_factor;
        }

        /// \brief Set the tau linear scaling factor.
        void setTau(const Scalar tau)
        {
            this->tau = tau;
        }
        /// \brief Get the tau linear scaling factor.
        Scalar getTau() const
        {
            return tau;
        }

        /// \brief Set the proximal value.
        void setProximalValue(const Scalar mu)
        {
            this->mu_prox = mu;
        }
        /// \brief Get the proximal value.
        Scalar getProximalValue() const
        {
            return mu_prox;
        }

        /// \brief Check if the solver is initialized
        bool isInitialized() const
        {
            return is_initialized;
        }

        void reset()
        {
            stats.reset();
            is_initialized = false;
            // num_iterations_ = 0;
            // primal_solution_.setZero();
            // dual_solution_.setZero();
            x_.setZero();
            y_.setZero();
            z_.setZero(); // Or gs ?
        }

        /// \brief Set the primal/dual ratio.
        void setRatioPrimalDual(const Scalar ratio_primal_dual)
        {
            PINOCCHIO_CHECK_INPUT_ARGUMENT(ratio_primal_dual > 0., "The ratio primal/dual should be positive strictly");
            this->ratio_primal_dual = ratio_primal_dual;
        }
        /// \brief Get the primal/dual ratio.
        Scalar getRatioPrimalDual() const
        {
            return ratio_primal_dual;
        }
        /// \brief Set the maximum number of Delassus decomposition updates.
        void setMaxDelassusDecompositionUpdates(const int max_updates)
        {
            this->max_delassus_decomposition_updates = max_updates;
        }
        /// \brief Get the maximum number of Delassus decomposition updates.
        int getMaxDelassusDecompositionUpdates() const
        {
            return max_delassus_decomposition_updates;
        }

        ///

        ///  \returns the number of updates of the Cholesky factorization due to rho updates.
        int getCholeskyUpdateCount() const
        {
            return cholesky_update_count;
        }

        /// \brief Set the dual momentum.
        void setDualMomentum(const Scalar dual_momentum)
        {
            this->dual_momentum = dual_momentum;
        }
        /// \brief Get the dual momentum.
        Scalar getDualMomentum() const
        {
            return dual_momentum;
        }

        /// \brief Set the rho momentum.
        void setRhoMomentum(const Scalar rho_momentum)
        {
            this->rho_momentum = rho_momentum;
        }
        /// \brief Get the rho momentum.
        Scalar getRhoMomentum() const
        {
            return rho_momentum;
        }
        /// \brief Set the rho update ratio.
        void setRhoUpdateRatio(const Scalar rho_update_ratio)
        {
            this->rho_update_ratio = rho_update_ratio;
        }
        /// \brief Get the rho update ratio.
        Scalar getRhoUpdateRatio() const
        {
            return rho_update_ratio;
        }

        /// \brief Set the rho minimum update frequency.
        void setRhoMinUpdateFrequency(const int rho_min_update_frequency)
        {
            this->rho_min_update_frequency = rho_min_update_frequency;
        }
        /// \brief Get the rho minimum update frequency.
        int getRhoMinUpdateFrequency() const
        {
            return rho_min_update_frequency;
        }

        /// \brief Sets the size of triangular matrix of Lanczos decomposition.
        /// The higher the size, the more accurate the estimation of min/max eigenvalues will be.
        /// Note: the maximum size is the size of the problem
        void setLanczosSize(int decomposition_size)
        {
            // TODO(lmontaut): should we throw if size > problem_size or instead take the min as done
            // below?
            int new_lanczos_size = pinocchio::math::max(2, this->problem_size);
            int new_lanczos_decomposition_size = pinocchio::math::max(2, pinocchio::math::min(decomposition_size, this->problem_size));
            if (new_lanczos_size != this->lanczos_decomposition.size()
                || new_lanczos_decomposition_size != this->lanczos_decomposition.decompositionSize())
            {
                this->lanczos_decomposition = LanczosDecomposition(
                    static_cast<Eigen::DenseIndex>(new_lanczos_size), static_cast<Eigen::DenseIndex>(new_lanczos_decomposition_size));
            }
        }

        /// \returns the Lanczos algorithm used for eigenvalues estimation.
        const LanczosDecomposition & getLanczosDecomposition() const
        {
            return lanczos_decomposition;
        }

        ADMMSolverStats & getStats()
        {
            return stats;
        }

        ///
        /// \brief Solve the constraint problem composed of problem data (G,g,constraint_models) and
        /// starting from the initial guess.
        ///
        /// \param[in] G Symmetric PSD matrix representing the Delassus of the constraint problem.
        /// \param[in] g Free constraint acceleration or velicity associted with the constraint
        /// problem.
        /// \param[in] constraint_models Vector of constraints.
        /// \param[in] preconditioner Precondtionner of the problem.
        /// \param[in] primal_guess Optional initial guess of the primal solution (constrained
        /// forces).
        /// \param[in] dual_guess Optinal Initial guess of the dual solution (constrained velocities).
        /// \param[in] solve_ncp whether to solve the NCP (true) or CCP (false)
        /// \param[in] admm_update_rule update rule for ADMM (linear or spectral)
        /// \param[in] stat_record record solver metrics
        ///
        /// \returns True if the problem has converged.
        template<
            typename DelassusDerived,
            typename VectorLike,
            template<typename T> class Holder,
            typename ConstraintModel,
            typename ConstraintAllocator>
        bool solve(
            pinocchio::DelassusOperatorBase<DelassusDerived> & delassus,
            const Eigen::MatrixBase<VectorLike> & g,
            const std::vector<Holder<const ConstraintModel>, ConstraintAllocator> & constraint_models,
            const Scalar dt,
            const boost::optional<RefConstVectorXs> preconditioner = boost::none,
            const boost::optional<RefConstVectorXs> primal_guess = boost::none,
            const boost::optional<RefConstVectorXs> dual_guess = boost::none,
            const bool solve_ncp = true,
            const ADMMUpdateRule admm_update_rule = ADMMUpdateRule::SPECTRAL,
            const boost::optional<Scalar> rho_init = boost::none,
            const boost::optional<Scalar> mu_prox_init = boost::none,
            const bool stat_record = false);

        ///
        /// \brief Solve the constraint problem composed of problem data (G,g,constraint_models) and
        /// starting from the initial guess.
        ///
        /// \param[in] G Symmetric PSD matrix representing the Delassus of the constraint problem.
        /// \param[in] g Free constraint acceleration or velicity associted with the constraint
        /// problem.
        /// \param[in] constraint_models Vector of constraints.
        /// \param[in] preconditioner Precondtionner of the problem.
        /// \param[in] primal_guess Optional initial guess of the primal solution (constrained
        /// forces).
        /// \param[in] dual_guess Optinal Initial guess of the dual solution (constrained velocities).
        /// \param[in] solve_ncp whether to solve the NCP (true) or CCP (false)
        /// \param[in] admm_update_rule update rule for ADMM (linear or spectral)
        /// \param[in] stat_record record solver metrics
        ///
        /// \returns True if the problem has converged.
        template<typename DelassusDerived, typename VectorLike, typename ConstraintModel, typename ConstraintAllocator>
        bool solve(
            pinocchio::DelassusOperatorBase<DelassusDerived> & delassus,
            const Eigen::MatrixBase<VectorLike> & g,
            const std::vector<ConstraintModel, ConstraintAllocator> & constraint_models,
            const Scalar dt,
            const boost::optional<RefConstVectorXs> preconditioner = boost::none,
            const boost::optional<RefConstVectorXs> primal_guess = boost::none,
            const boost::optional<RefConstVectorXs> dual_guess = boost::none,
            const bool solve_ncp = true,
            const ADMMUpdateRule admm_update_rule = ADMMUpdateRule::SPECTRAL,
            const boost::optional<Scalar> rho_init = boost::none,
            const boost::optional<Scalar> mu_prox_init = boost::none,
            const bool stat_record = false)
        {
            typedef std::reference_wrapper<const ConstraintModel> WrappedConstraintModelType;
            typedef std::vector<WrappedConstraintModelType> WrappedConstraintModelVector;

            WrappedConstraintModelVector wrapped_constraint_models(constraint_models.cbegin(), constraint_models.cend());

            return solve(
                delassus, g, wrapped_constraint_models, dt, preconditioner, primal_guess, dual_guess, solve_ncp, admm_update_rule, rho_init,
                mu_prox_init, stat_record);
        }

        ///
        /// \brief Solve the constraint problem composed of problem data (G,g,constraint_models) and
        /// starting from the initial guess.
        ///
        /// \param[in] G Symmetric PSD matrix representing the Delassus of the constraint problem.
        /// \param[in] g Free constraint acceleration or velicity associted with the constraint
        /// problem.
        /// \param[in] constraint_models Vector of constraints.
        /// \param[in] primal_guess Optional initial guess of the primal solution (constrained
        /// forces).
        /// \param[in] solve_ncp whether to solve the NCP (true) or CCP (false)
        ///
        /// \returns True if the problem has converged.
        template<
            typename DelassusDerived,
            typename VectorLike,
            template<typename T> class Holder,
            typename ConstraintModel,
            typename ConstraintAllocator,
            typename VectorLikeOut>
        bool solve(
            pinocchio::DelassusOperatorBase<DelassusDerived> & delassus,
            const Eigen::MatrixBase<VectorLike> & g,
            const std::vector<Holder<const ConstraintModel>, ConstraintAllocator> & constraint_models,
            const Scalar dt,
            const Eigen::DenseBase<VectorLikeOut> & primal_guess,
            const bool solve_ncp = true)
        {
            return solve(
                delassus.derived(), g.derived(), constraint_models, dt, boost::none, primal_guess.derived(), boost::none, solve_ncp);
        }

        ///
        /// \brief Solve the constraint problem composed of problem data (G,g,constraint_models) and
        /// starting from the initial guess.
        ///
        /// \param[in] G Symmetric PSD matrix representing the Delassus of the constraint problem.
        /// \param[in] g Free constraint acceleration or velicity associted with the constraint
        /// problem.
        /// \param[in] constraint_models Vector of constraints.
        /// \param[in] primal_guess Optional initial guess of the primal solution (constrained
        /// forces).
        /// \param[in] solve_ncp whether to solve the NCP (true) or CCP (false)
        ///
        /// \returns True if the problem has converged.
        template<
            typename DelassusDerived,
            typename VectorLike,
            typename ConstraintModel,
            typename ConstraintAllocator,
            typename VectorLikeOut>
        bool solve(
            pinocchio::DelassusOperatorBase<DelassusDerived> & delassus,
            const Eigen::MatrixBase<VectorLike> & g,
            const std::vector<ConstraintModel, ConstraintAllocator> & constraint_models,
            const Scalar dt,
            const Eigen::DenseBase<VectorLikeOut> & primal_guess,
            const bool solve_ncp = true)
        {
            typedef std::reference_wrapper<const ConstraintModel> WrappedConstraintModelType;
            typedef std::vector<WrappedConstraintModelType> WrappedConstraintModelVector;

            WrappedConstraintModelVector wrapped_constraint_models(constraint_models.cbegin(), constraint_models.cend());

            return solve(delassus, g, wrapped_constraint_models, dt, primal_guess, solve_ncp);
        }

        /// \returns the primal solution of the problem
        const VectorXs & getPrimalSolution() const
        {
            return y_;
        }
        /// \returns the dual solution of the problem
        const VectorXs & getDualSolution() const
        {
            return z_constraint_;
        }
        /// \returns the complementarity shift
        const VectorXs & getComplementarityShift() const
        {
            return s_constraint_;
        }

        /// \returns the scaled primal solution of the problem
        const VectorXs & getScaledPrimalSolution() const
        {
            return y_bar_;
        }
        /// \returns the scaled dual solution of the problem
        const VectorXs & getScaledDualSolution() const
        {
            return z_bar_;
        }
        /// \returns the scaled complementarity shift
        const VectorXs & getScaledComplementarityShift() const
        {
            return s_bar_;
        }

        /// \returns use the preconditioner to scale a primal quantity x.
        /// Typically, it allows to get x_bar from x.
        void scalePrimalSolution(const VectorXs & x, VectorXs & x_bar) const
        {
            preconditioner_.scale(x, x_bar);
        }

        /// \returns use the preconditioner to unscale a primal quantity x.
        /// Typically, it allows to get x from x_bar.
        void unscalePrimalSolution(const VectorXs & x_bar, VectorXs & x) const
        {
            preconditioner_.unscale(x_bar, x);
        }

        /// \returns use the preconditioner to scale a dual quantity z.
        /// Typically, it allows to get z_bar from z.
        void scaleDualSolution(const VectorXs & z, VectorXs & z_bar) const
        {
            preconditioner_.unscale(z, z_bar);
        }

        /// \returns use the preconditioner to unscale a dual quantity z.
        /// Typically, it allows to get z from z_bar.
        void unscaleDualSolution(const VectorXs & z_bar, VectorXs & z) const
        {
            preconditioner_.scale(z_bar, z);
        }

    protected:
        bool is_initialized;

        /// \brief proximal value
        Scalar mu_prox;

        /// \brief Linear scaling of the ADMM penalty term
        Scalar tau;
        /// \brief Penalty term associated to the ADMM.
        Scalar rho;

        // Set of parameters associated with the Spectral update rule
        /// \brief Power value associated to rho. This quantity will be automatically updated.
        Scalar rho_power;
        /// \brief Update factor for the primal/dual update of rho.
        Scalar rho_power_factor;

        // Set of parameters associated with the Linear update rule
        /// \brief value of the increase/decrease factor
        Scalar linear_update_rule_factor;

        ///  \brief Ratio primal/dual
        Scalar ratio_primal_dual;

        /// \brief Maximum number of Delassus decomposition updates
        int max_delassus_decomposition_updates;
        /// \brief Momentum for dual variable update
        Scalar dual_momentum;
        /// \brief Momentum for rho update
        Scalar rho_momentum;

        /// \brief Ratio for rho update
        Scalar rho_update_ratio;
        /// \brief Minimum frequency (in number of iterations) for rho update
        int rho_min_update_frequency;

        /// \brief Lanczos decomposition algorithm.
        LanczosDecomposition lanczos_decomposition;

        /// \brief Primal variables (corresponds to the constraint impulses)
        VectorXs x_, y_;
        /// \brief Scaled primal variables (corresponds to the contact forces)
        VectorXs x_bar_, y_bar_;
        /// \brief Previous values of x_bar_, y_bar_ and z_bar_.
        VectorXs x_bar_previous, y_bar_previous, z_bar_previous;
        /// \brief Dual variable of the ADMM (corresponds to the contact velocity or acceleration).
        VectorXs z_;
        VectorXs z_constraint_;
        /// \brief Scaled dual variable of the ADMM (corresponds to the contact velocity or
        /// acceleration).
        VectorXs z_bar_;
        /// \brief De Saxé shift
        VectorXs s_;
        VectorXs s_constraint_;
        /// \brief Scaled De Saxé shift
        VectorXs s_bar_;

        /// \brief the diagonal preconditioner of the problem
        DiagonalPreconditioner preconditioner_;
        /// \brief Preconditioned drift term
        VectorXs g_bar_;

        /// \brief Time scaling vector for constraints
        VectorXs time_scaling_acc_to_constraints, time_scaling_constraints_to_pos;
        /// \brief Vector g divided by time scaling (g / time_scaling_acc_to_constraints)
        VectorXs gs;

        VectorXs rhs, primal_feasibility_vector, primal_feasibility_vector_bar, dual_feasibility_vector, dual_feasibility_vector_bar;

        int cholesky_update_count;

        ADMMSolverStats stats;

#ifdef PINOCCHIO_WITH_HPP_FCL
        using Base::timer;
#endif // PINOCCHIO_WITH_HPP_FCL
    }; // struct ADMMContactSolverTpl

} // namespace simplex

// Include implementation
#include "simplex/solver/admm-solver.hxx"

#endif // ifndef __simplex_solver_admm_solver_hpp__
