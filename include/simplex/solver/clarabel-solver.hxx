#ifndef __simplex_solver_clarabel_solver_hxx__
#define __simplex_solver_clarabel_solver_hxx__

#include "simplex/solver/clarabel-solver.hpp"

#include <pinocchio/macros.hpp>
#include <pinocchio/algorithm/constraints/constraints.hpp>
#include <Eigen/Core>

namespace simplex
{
    namespace internal
    {
        template<
            typename ConstraintModel,          //
            typename ConstraintModelAllocator, //
            typename ConstraintData,           //
            typename ConstraintDataAllocator,  //
            typename VectorInLike,             //
            typename VectorOutLike>
        void computeDeSaxceShift(
            const std::vector<ConstraintModel, ConstraintModelAllocator> & constraint_models,
            const std::vector<ConstraintData, ConstraintDataAllocator> & constraint_datas,
            const Eigen::MatrixBase<VectorInLike> & velocity,
            const Eigen::MatrixBase<VectorOutLike> & shift_)
        {
            using VectorInLikeInnerType = typename std::decay_t<VectorInLike>::PlainObject;
            using Scalar = typename VectorInLikeInnerType::Scalar;
            enum
            {
                Options = VectorInLikeInnerType::Options
            };
            using Vector3s = Eigen::Matrix<Scalar, 3, 1>;
            using pinocchio::helper::get_ref;

            PINOCCHIO_UNUSED_VARIABLE(constraint_datas);

            VectorOutLike & shift = shift_.const_cast_derived();
            const auto nc = static_cast<Eigen::Index>(constraint_models.size());
            Eigen::Index idx = 0;
            for (Eigen::Index i = 0; i < nc /*num constraints*/; ++i)
            {
                const auto & cmodel = get_ref(constraint_models[static_cast<std::size_t>(i)]);
                // const auto & cdata = get_ref(constraint_datas[static_cast<std::size_t>(i)]);
                using ContactModel = ::pinocchio::FrictionalPointConstraintModelTpl<Scalar, Options>;
                // using ContactData = ::pinocchio::FrictionalPointConstraintDataTpl<Scalar, Options>;

                if (const ContactModel * fpc_model = boost::get<const ContactModel>(&cmodel))
                {
                    // const ContactData * fpc_data = boost::get<const ContactData>(&cdata);
                    const Scalar mu = fpc_model->set().mu;
                    const Scalar vt_norm = velocity.template segment<2>(idx + 1).norm();
                    shift.template segment<3>(idx) = Vector3s(mu * vt_norm, 0, 0);
                    idx += 3;
                }
                else
                {
                    PINOCCHIO_THROW_PRETTY(std::runtime_error, "Constraint not supported yet");
                }
            }
        }
    } // namespace internal

    template<typename Scalar, int Options>
    template<typename DelassusOperator, typename VectorLike, typename ConstraintModelAllocator, typename ConstraintDataAllocator>
    void ClarabelContactSolverTpl<Scalar, Options>::solve(
        const DelassusOperator & delassus,
        const Eigen::MatrixBase<VectorLike> & g,
        const std::vector<std::reference_wrapper<const ::pinocchio::ConstraintModelTpl<Scalar, Options>>, ConstraintModelAllocator> &
            constraint_models,
        const std::vector<std::reference_wrapper<::pinocchio::ConstraintDataTpl<Scalar, Options>>, ConstraintDataAllocator> &
            constraint_datas,
        const boost::optional<Eigen::Ref<const VectorXs>> & preconditioner,
        const boost::optional<Eigen::Ref<const VectorXs>> & primal_guess,
        const boost::optional<Eigen::Ref<const VectorXs>> & dual_guess,
        bool is_ncp,
        bool record_stats)
    {
        PINOCCHIO_UNUSED_VARIABLE(constraint_datas);

        assert(g.size() == problem_size_ && "Drift vector size mismatch");

        // Resize solution vectors if needed
        if (primal_solution_.size() != problem_size_)
        {
            primal_solution_.resize(problem_size_);
            dual_solution_.resize(problem_size_);
        }

        // Build problem matrices for Clarabel
        // The contact problem is: min (1/2) lambda^T G lambda + g^T lambda
        //                         s.t. lambda in K (cone constraints)
        // In Simple, the convention is that the second-order cone is {(x, y, z), sqrt(x^2 + y^2) <= z}.
        // In Clarabel, the second-order cone convention is {(x, y, z), sqrt(y^2 + z^2) <= x}
        // So we have to do a permutation.
        // Get the dense Delassus matrix and the free velocity g
        using MapVectorXs = Eigen::Map<VectorXs>;
        using MapMatrixXs = Eigen::Map<MatrixXs>;
        MapMatrixXs G_dense(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, problem_size_, problem_size_));
        MapVectorXs q(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, problem_size_, 1));
        MapVectorXs q_new(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, problem_size_, 1));
        MapVectorXs b(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, problem_size_, 1));
        MapVectorXs frictions(PINOCCHIO_EIGEN_MAP_ALLOCA(Scalar, problem_size_, 1));
        delassus.matrix(G_dense, true); // true = enforce symmetry
        // permutate columns
        for (Eigen::Index i = 0; i < G_dense.cols(); i += 3)
        {
            G_dense.col(i).swap(G_dense.col(i + 1));
            G_dense.col(i).swap(G_dense.col(i + 2));
        }
        // permutate rows
        for (Eigen::Index i = 0; i < G_dense.rows(); i += 3)
        {
            G_dense.row(i).swap(G_dense.row(i + 1));
            G_dense.row(i).swap(G_dense.row(i + 2));
        }
        // Convert to sparse format for Clarabel
        // SparseMatrix P(G_dense.sparseView()); // = G_dense.sparseView();
        SparseMatrix P(G_dense.sparseView()); // = G_dense.sparseView();
        q = g;
        for (Eigen::Index i = 0; i < q.size(); i += 3)
        {
            std::swap(q.coeffRef(i), q.coeffRef(i + 1));
            std::swap(q.coeffRef(i), q.coeffRef(i + 2));
        }

        // Build cones and constraint matrix A and vector b
        // For the contact problem formulation, we have:
        // A * lambda + s = b, where s in K (friction cone)
        // We encode the friction coefficient in the diagonal A matrix
        std::vector<clarabel::SupportedConeT<Scalar>> cones;
        cones.reserve(constraint_models.size());
        b.setZero();
        const Eigen::Index nc = static_cast<Eigen::Index>(constraint_models.size());
        for (Eigen::Index i = 0; i < nc; ++i)
        {
            const auto & cmodel = constraint_models[static_cast<std::size_t>(i)].get();
            const auto & cdata = constraint_datas[static_cast<std::size_t>(i)].get();
            using ConstraintModel = ::pinocchio::FrictionalPointConstraintModelTpl<Scalar, Options>;
            // using ConstraintData = ::pinocchio::FrictionalPointConstraintDataTpl<Scalar, Options>;
            if (const ConstraintModel * fpc_model = boost::get<const ConstraintModel>(&cmodel))
            {
                // const ConstraintData * fpc_data = boost::get<const ConstraintData>(&cdata);
                frictions.template segment<3>(3 * i) = -Vector3s(fpc_model->set().mu, 1, 1);
                cones.emplace_back(clarabel::SecondOrderConeT<Scalar>(3));
            }
            else
            {
                PINOCCHIO_THROW_PRETTY(std::runtime_error, "Constraint not supported yet");
            }
        }
        //
        // Apply permutation to friction coefficients to match Clarabel's cone convention
        SparseMatrix A(problem_size_, problem_size_);
        A.reserve(Eigen::VectorXi::Constant(problem_size_, 1));
        for (Eigen::Index i = 0; i < problem_size_; ++i)
        {
            const Scalar value = frictions(i);
            A.coeffRef(i, i) = value;
        }
        A.makeCompressed();

        // Create Clarabel settings
        clarabel::DefaultSettings<Scalar> settings = clarabel::DefaultSettings<Scalar>::default_settings();
        settings.max_iter = static_cast<uint32_t>(max_iterations_);
        settings.tol_gap_abs = absolute_tolerance_;
        settings.tol_gap_rel = relative_tolerance_;
        settings.tol_feas = absolute_tolerance_;
        settings.tol_infeas_abs = absolute_tolerance_;
        settings.tol_infeas_rel = relative_tolerance_;
        settings.tol_ktratio = relative_tolerance_;
        settings.time_limit = 100000;

        settings.verbose = false;

        // Create the Clarabel solver
        clarabel_solver_ = std::make_shared<clarabel::DefaultSolver<Scalar>>(P, q, A, b, cones, settings);

        // Solve the problem
        clarabel_solver_->solve();
        num_iterations_ = static_cast<int>(clarabel_solver_->info().iterations);
        prev_x_ = clarabel_solver_->solution().x;
        prev_z_ = clarabel_solver_->solution().z;
        internal::computeDeSaxceShift(constraint_models, constraint_datas, prev_z_, shift_);

        bool match = !is_ncp || (shift_.norm() <= absolute_tolerance_);
        const int max_ncp_loops = 25;
        int ncp_loops = 1;
        prev_shift_ = shift_;
        // if (!match)
        // {
        //   std::cout << "\n";
        // }
        while (!match && ncp_loops <= max_ncp_loops)
        {
            q_new = q + prev_shift_;
            clarabel_solver_->update_q(q_new);
            clarabel_solver_->solve();
            num_iterations_ += static_cast<int>(clarabel_solver_->info().iterations);

            const clarabel::DefaultSolution<Scalar> & solution = clarabel_solver_->solution();
            internal::computeDeSaxceShift(constraint_models, constraint_datas, solution.z, shift_);
            // if ((prev_x_ - solution.x).norm() <= absolute_tolerance_ || (prev_z_ - solution.z).norm() <= absolute_tolerance_)
            if ((shift_ - prev_shift_).norm() <= absolute_tolerance_) // NCP measure
            {
                match = true;
                // std::cout << "-----> Solved NCP in " << ncp_loops << " outer loops\n";
            }
            else
            {
                // std::cout << "(solution.x - prev_x_).norm() = " << (solution.x - prev_x_).norm();
                // std::cout << " / (solution.z - prev_z_).norm() = " << (solution.z - prev_z_).norm();
                // std::cout << " / (prev_shift_ - shift_).norm() = " << (prev_shift_ - shift_).norm();
                // std::cout << " / additional iters = " << clarabel_solver_->info().iterations << "\n";
                prev_x_ = solution.x;
                prev_z_ = solution.z;
                prev_shift_ = shift_;
            }
            ++ncp_loops;
        }

        // Extract solution
        const clarabel::DefaultSolution<Scalar> & solution = clarabel_solver_->solution();

        for (Eigen::Index i = 0; i < solution.x.size(); i += 3)
        {
            primal_solution_(i) = solution.x(i + 1);
            primal_solution_(i + 1) = solution.x(i + 2);
            primal_solution_(i + 2) = solution.x(i);
            //
            dual_solution_(i) = solution.z(i + 1);
            dual_solution_(i + 1) = solution.z(i + 2);
            dual_solution_(i + 2) = solution.z(i);
        }

        // Get solver info
        num_iterations_ += static_cast<int>(clarabel_solver_->info().iterations);

        // Suppress unused parameter and variable warnings
        (void)constraint_datas;
        (void)preconditioner;
        (void)primal_guess;
        (void)dual_guess;
        (void)record_stats;

        is_initialized_ = true;
    }

} // namespace simplex

#endif // __simplex_solver_clarabel_solver_hxx__
