#ifndef __simplex_core_ncp_derivatives_hxx__
#define __simplex_core_ncp_derivatives_hxx__

#include "simplex/core/ncp-derivatives.hpp"
#include "simplex/math/qr.hpp"
#include "simplex/tracy.hpp"

namespace simplex
{
    // --------------------------------------------------------------------------
    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void ContactSolverDerivativesTpl<S, O, JointCollectionTpl>::allocate()
    {
        const size_t num_max_contacts = constraint_problem().getMaxNumberOfContacts();
        const auto max_problem_size = (Eigen::Index)(3 * num_max_contacts);
        G_data_.resize(max_problem_size * max_problem_size);
        delassus_data_.resize(max_problem_size * max_problem_size);
        E_.reserve(num_max_contacts);
        ETP_.reserve(num_max_contacts);
        rhs_data_.resize(max_problem_size * max_theta_dim_);
        dlam_dtheta_data_.resize(max_problem_size * max_theta_dim_);
        dlam_dtheta_reduced_data_.resize(max_problem_size * max_theta_dim_);
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void ContactSolverDerivativesTpl<S, O, JointCollectionTpl>::buildSystemOfImplicitGrads()
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::buildSystemOfImplicitGrads");

        const ContactIndexVector & sticking_contacts = constraint_problem().sticking_contacts;
        const ContactIndexVector & sliding_contacts = constraint_problem().sliding_contacts;
        const auto n_stk = (Eigen::Index)(sticking_contacts.size());
        const auto n_sld = (Eigen::Index)(sliding_contacts.size());

        // Computing necesary quantities
        E_.clear();
        ETP_.clear();
        const Vector3s e_z(0, 0, Scalar(1));
        for (Eigen::Index i = 0; i < n_sld; ++i)
        {
            const auto sld_contact_id = (Eigen::Index)(sliding_contacts[(std::size_t)(i)]);
            const Vector3s lambda_i = constraint_problem().frictional_point_constraints_forces().template segment<3>(3 * sld_contact_id);
            const Vector3s time_scaling_i =
                constraint_problem().contact_time_scaling_acc_to_constraints().template segment<3>(3 * sld_contact_id);
            const Vector3s sigma_i =
                constraint_problem().frictional_point_constraints_velocities().template segment<3>(3 * sld_contact_id).array()
                / time_scaling_i.array();
            // Compute the basis of the plane tangent to the friction cone
            Matrix32s Ebasis;
            Ebasis.col(0) = lambda_i.normalized();
            assert(pinocchio::math::fabs(lambda_i.norm()) > 1e-15 && "Loss of numerical precision.");
            Vector3s u_T = Vector3s::Zero();
            u_T.template head<2>() = sigma_i.template head<2>().normalized();
            Ebasis.col(1) = e_z.cross(u_T);
            assert(pinocchio::math::fabs(sigma_i.norm()) > 1e-15 && "Loss of numerical precision.");
            E().push_back(Ebasis);

            const Scalar lambda_n = lambda_i[2];
            const ConstraintModel & cmodel = constraint_problem().constraint_models[(size_t)(sld_contact_id)];
            if (const FrictionalPointConstraintModel * fpcmodel = boost::get<const FrictionalPointConstraintModel>(&cmodel))
            {
                const Scalar mu = fpcmodel->set().mu;
                const Scalar alpha_inv = (mu * lambda_n) / sigma_i.template head<2>().norm();
                assert(pinocchio::math::fabs(sigma_i.template head<2>().norm()) > 1e-15 && "Loss of numerical precision.");

                Matrix23s ETPbasis = Matrix23s::Zero();
                ETPbasis(0, 2) = Ebasis(2, 0);
                ETPbasis.row(1).template head<2>() = Ebasis.col(1).template head<2>().transpose();
                ETPbasis.row(1).template head<2>() *= alpha_inv;
                if (!constraint_problem().is_ncp)
                {
                    ETPbasis.row(0).template head<2>() += -mu * Ebasis(2, 0) * u_T.template head<2>();
                }
                ETP().push_back(ETPbasis);
            }
            else
            {
                // TODO(quentinll) for now we only handle contact constraints
            }
        }
        gradients_problem_size_ = (int)(3 * n_stk + 2 * n_sld);

        // Computing lines corresponding to sticking contacts
        for (Eigen::Index i = 0; i < n_stk; ++i)
        {
            const auto stk_contact_id = (Eigen::Index)(sticking_contacts[(std::size_t)(i)]);

            for (Eigen::Index j = 0; j < n_stk; ++j)
            {
                const auto stk_contact_id2 = (Eigen::Index)(sticking_contacts[(std::size_t)(j)]);
                G().template block<3, 3>(3 * i, 3 * j) = delassus().template block<3, 3>(3 * stk_contact_id, 3 * stk_contact_id2);
            }

            for (Eigen::Index j = 0; j < n_sld; ++j)
            {
                const auto sld_contact_id = (Eigen::Index)(sliding_contacts[(std::size_t)(j)]);
                G().template block<3, 2>(3 * i, 3 * n_stk + 2 * j).noalias() =
                    delassus().template block<3, 3>(3 * stk_contact_id, 3 * sld_contact_id) * E()[size_t(j)];
            }
        }

        // Computing lines corresponding to sliding contacts
        for (Eigen::Index i = 0; i < n_sld; ++i)
        {
            const auto sld_contact_id = (Eigen::Index)(sliding_contacts[(std::size_t)(i)]);

            for (Eigen::Index j = 0; j < n_stk; ++j)
            {
                const auto stk_contact_id = (Eigen::Index)(sticking_contacts[(std::size_t)(j)]);
                G().template block<2, 3>(3 * n_stk + 2 * i, 3 * j).noalias() =
                    E()[size_t(i)].transpose() * delassus().template block<3, 3>(3 * sld_contact_id, 3 * stk_contact_id);
            }

            for (Eigen::Index j = 0; j < n_sld; ++j)
            {
                const auto sld_contact_id2 = (Eigen::Index)(sliding_contacts[(std::size_t)(j)]);
                G().template block<2, 2>(3 * n_stk + 2 * i, 3 * n_stk + 2 * j).noalias() =
                    ETP()[size_t(i)] * delassus().template block<3, 3>(3 * sld_contact_id, 3 * sld_contact_id2) * E()[size_t(j)];
            }

            G()(3 * n_stk + 2 * i + 1, 3 * n_stk + 2 * i + 1) += Scalar(1);
        }
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    void ContactSolverDerivativesTpl<S, O, JointCollectionTpl>::compute()
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::compute");

        constraint_problem().collectActiveSet();

        if (constraint_problem().sticking_contacts.size() == 0 && constraint_problem().sliding_contacts.size() == 0)
            return;

        // TODO for now we only consider the conatct forces and not other constraints.
        // if the system contains constraints during the forward simulation, gradients will be wrong.
        constraint_problem().constraint_cholesky_decomposition.getDelassusCholeskyExpression().matrix(delassus());
        delassus() -= constraint_problem().constraint_cholesky_decomposition.getDamping().asDiagonal(); // true delassus
        delassus() =
            delassus().bottomRightCorner(constraint_problem().constraints_problem_size(), constraint_problem().constraints_problem_size());

        buildSystemOfImplicitGrads();

        // Computing decomposition of the implicit gradient matrix
        switch (implicit_gradient_solver_type)
        {
        case ImplicitGradientSystemSolver::COD: {
            SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::compute - Compute COD decomposition");
            using SolverType = Eigen::CompleteOrthogonalDecomposition<Eigen::Ref<MatrixXs>>;
            cod_solver_ = std::make_shared<SolverType>(G().derived());
        }
        break;
        case ImplicitGradientSystemSolver::HOUSEHOLDER_QR: {
            SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::compute - Compute QR decomposition");
            using SolverType = Eigen::HouseholderQR<Eigen::Ref<MatrixXs>>;
            qr_solver_ = std::make_shared<SolverType>(G().derived());
        }
        break;
        }
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<typename Matrix>
    void ContactSolverDerivativesTpl<S, O, JointCollectionTpl>::jvp(const Eigen::MatrixBase<Matrix> & dGlamgdtheta)
    {
        SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::jvp");

        if (measure_timings)
        {
            timer_.start();
        }

        theta_size_ = dGlamgdtheta.cols();
        assert(theta_size_ <= max_theta_dim_ && "The number of columns of dGlamgdtheta is too large.");
        assert(
            dGlamgdtheta.rows() == (int)(3 * (constraint_problem().getNumberOfContacts()))
            && "The number of rows of dGlamgdtheta is not equal to "
               "3*constraint_problem.getNumberOfContacts().");

        PINOCCHIO_EIGEN_MALLOC_NOT_ALLOWED();

        dlam_dtheta().setZero();
        if (constraint_problem().sticking_contacts.size() != 0 || constraint_problem().sliding_contacts.size() != 0)
        { // There is at least one sliding or sticking contact

            expressInMinimalSystem(dGlamgdtheta.derived(), rhs());

            // solving the reduced system on implicit gradients
            {
                // #ifndef NDEBUG
                //         const MatrixXs G_copy(G()); //TODO: should be copied before the
                //         decomposition
                // #endif

                switch (implicit_gradient_solver_type)
                {
                case ImplicitGradientSystemSolver::COD: {
                    using SolverType = Eigen::CompleteOrthogonalDecomposition<Eigen::Ref<MatrixXs>>;
                    using SolveInPlaceWrapper = math::SolveInPlaceWrapper<SolverType>;
                    auto & solver_wrapper = reinterpret_cast<SolveInPlaceWrapper &>(*(cod_solver_));
                    {
                        SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::jvp - Solve COD");
                        dlam_dtheta_reduced() = rhs();
                        solver_wrapper.solveInPlace(dlam_dtheta_reduced());
                    }
                }
                break;
                case ImplicitGradientSystemSolver::HOUSEHOLDER_QR: {
                    using SolverType = Eigen::HouseholderQR<Eigen::Ref<MatrixXs>>;
                    using SolveInPlaceWrapper = math::SolveInPlaceWrapper<SolverType>;
                    auto & solver_wrapper = reinterpret_cast<SolveInPlaceWrapper &>(*(qr_solver_));
                    {
                        SIMPLEX_TRACY_ZONE_SCOPED_N("ContactSolverDerivatives::jvp - Solve QR");
                        dlam_dtheta_reduced() = rhs();
                        solver_wrapper.solveInPlace(dlam_dtheta_reduced());
                    }
                }
                break;
                }

                // TODO(jcarpent): add C++ test to check that G_copy *
                // dlam_dtheta_reduced()).isApprox(rhs())
            }

            // retrieves gradients from the reduced solution
            extractFromMinimalSystem(dlam_dtheta_reduced(), dlam_dtheta());
        }

        PINOCCHIO_EIGEN_MALLOC_ALLOWED();

        if (measure_timings)
        {
            timer_.stop();
            contact_solver_derivatives_timings_ = timer_.elapsed();
        }
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<typename MatrixIn, typename MatrixOut>
    void ContactSolverDerivativesTpl<S, O, JointCollectionTpl>::expressInMinimalSystem(
        const Eigen::MatrixBase<MatrixIn> & rhs, const Eigen::MatrixBase<MatrixOut> & reduced_rhs_) const
    {
        auto & reduced_rhs = reduced_rhs_.const_cast_derived();

        const ContactIndexVector & sticking_contacts = constraint_problem().sticking_contacts;
        const ContactIndexVector & sliding_contacts = constraint_problem().sliding_contacts;
        const auto n_stk = (Eigen::Index)(sticking_contacts.size());
        const auto n_sld = (Eigen::Index)(sliding_contacts.size());

        // Computing lines corresponding to sticking contacts
        for (Eigen::Index i = 0; i < n_stk; ++i)
        {
            const auto stk_contact_id = (Eigen::Index)(sticking_contacts[(std::size_t)(i)]);

            reduced_rhs.template middleRows<3>(3 * i) = -rhs.template middleRows<3>(3 * stk_contact_id);
        }

        // Computing lines corresponding to sliding contacts
        for (Eigen::Index i = 0; i < n_sld; ++i)
        {
            const auto sld_contact_id = (Eigen::Index)(sliding_contacts[(std::size_t)(i)]);

            reduced_rhs.template middleRows<2>(3 * n_stk + 2 * i).noalias() =
                -ETP()[size_t(i)] * rhs.template middleRows<3>(3 * sld_contact_id);
        }
    }

    template<typename S, int O, template<typename, int> class JointCollectionTpl>
    template<typename MatrixIn, typename MatrixOut>
    void ContactSolverDerivativesTpl<S, O, JointCollectionTpl>::extractFromMinimalSystem(
        const Eigen::MatrixBase<MatrixIn> & reduced_sol, const Eigen::MatrixBase<MatrixOut> & dlambda_dtheta_) const
    {
        auto & dlambda_dtheta = dlambda_dtheta_.const_cast_derived();

        const ContactIndexVector & sticking_contacts = constraint_problem().sticking_contacts;
        const ContactIndexVector & sliding_contacts = constraint_problem().sliding_contacts;
        const auto n_stk = (Eigen::Index)(sticking_contacts.size());
        const auto n_sld = (Eigen::Index)(sliding_contacts.size());

        for (Eigen::Index i = 0; i < n_stk; ++i)
        {
            const auto stk_contact_id = (Eigen::Index)(sticking_contacts[(std::size_t)(i)]);
            dlambda_dtheta.template middleRows<3>(3 * stk_contact_id) = reduced_sol.template middleRows<3>(3 * i);
        }

        for (Eigen::Index i = 0; i < n_sld; ++i)
        {
            const auto sld_contact_id = (Eigen::Index)(sliding_contacts[(std::size_t)(i)]);
            dlambda_dtheta.template middleRows<3>(3 * sld_contact_id).noalias() =
                E()[size_t(i)] * reduced_sol.template middleRows<2>(3 * n_stk + 2 * i);
        }
    }

} // namespace simplex

#endif // ifndef __simplex_core_ncp_derivatives_hxx__
