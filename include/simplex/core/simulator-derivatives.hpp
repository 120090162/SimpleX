#ifndef __simplex_core_simulator_derivatives_hpp__
#define __simplex_core_simulator_derivatives_hpp__

#include "simplex/core/fwd.hpp"
#include "simplex/macros.hpp"
#include "simplex/core/simulator-x.hpp"
#include "simplex/core/constraints-problem-derivatives.hpp"
#include "simplex/core/contact-frame.hpp"
#include "simplex/core/ncp-derivatives.hpp"

#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace simplex
{

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct traits<SimulatorDerivativesTpl<_Scalar, _Options, JointCollectionTpl>>
    {
        using Scalar = _Scalar;
    };

    template<typename _Scalar, int _Options, template<typename, int> class JointCollectionTpl>
    struct SimulatorDerivativesTpl
    {
        // TODO: template by allocator
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Typedefs
        using ConstraintsProblemDerivatives = ConstraintsProblemDerivativesTpl<_Scalar, _Options, JointCollectionTpl>;
        using ContactSolverDerivatives = ContactSolverDerivativesTpl<_Scalar, _Options, JointCollectionTpl>;
        using SimulatorX = SimulatorXTpl<_Scalar, _Options, JointCollectionTpl>;
        using PlacementFromNormalAndPosition = PlacementFromNormalAndPositionTpl<_Scalar, _Options>;
        using Scalar = _Scalar;
        enum
        {
            Options = _Options
        };

        using GeometryObject = ::pinocchio::GeometryObject;
        using CollisionResult = ::coal::CollisionResult;
        using ContactPatchResult = ::coal::ContactPatchResult;
        using ContactPatch = ::coal::ContactPatch;

        using GeomIndex = typename ContactSolverDerivatives::GeomIndex;
        using ContactIndex = typename ContactSolverDerivatives::ContactIndex;
        using JointIndex = typename ContactSolverDerivatives::JointIndex;

        using VectorXs = typename ContactSolverDerivatives::VectorXs;
        using Vector3s = typename ContactSolverDerivatives::Vector3s;
        using Vector2s = typename ContactSolverDerivatives::Vector2s;
        using Matrix23s = typename ContactSolverDerivatives::Matrix23s;

        using MatrixXs = typename ContactSolverDerivatives::MatrixXs;
        using Matrix63s = Eigen::Matrix<Scalar, 6, 3, Options>;
        using Matrix36s = Eigen::Matrix<Scalar, 3, 6, Options>;
        using Matrix6s = Eigen::Matrix<Scalar, 6, 6, Options>;
        using Matrix6Xs = Eigen::Matrix<Scalar, 6, Eigen::Dynamic, Options>;
        using RowMatrixXs = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Options | Eigen::RowMajor>;

        using FrictionalPointConstraintModel = typename ConstraintsProblemDerivatives::FrictionalPointConstraintModel;
        using ConstraintModel = typename ConstraintsProblemDerivatives::ConstraintModel;

        using SE3 = typename SimulatorX::SE3;
        using Motion = typename SimulatorX::Motion;
        using Force = typename SimulatorX::Force;

        using Data = typename SimulatorX::Data;

        using ContactMapper = typename ConstraintsProblemDerivatives::ContactMapper;
        using ConstraintCholeskyDecomposition = ::pinocchio::ContactCholeskyDecompositionTpl<Scalar, Options>;

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
        using ContactPatchDerivativeRequest = ::diffcoal::ContactPatchDerivativeRequest;
        using ContactPatchDerivative = ::diffcoal::ContactPatchDerivative;
#endif

        /// \brief number of contacts
        ContactIndex nc;

        /// \brief Whether or not timings of the `stepDerivatives` function are measured.
        /// If set to true, the timing of the last call to `stepDerivatives` can be accessed via
        /// `getSimulatorDerivativesCPUTimes`
        bool measure_timings{false};

        /// \brief Jacobian of the velocity from the step function wrt q.
        RowMatrixXs dvnew_dq;

        /// \brief Jacobian of the velocity from the step function wrt v.
        RowMatrixXs dvnew_dv;

        /// \brief Jacobian of the velocity from the step function wrt tau.
        RowMatrixXs dvnew_dtau;

        /// \brief Jacobian of the acceleration from the step function wrt q.
        RowMatrixXs danew_dq;

        /// \brief Jacobian of the acceleration from the step function wrt v.
        RowMatrixXs danew_dv;

        /// \brief Jacobian of the acceleration from the step function wrt tau.
        RowMatrixXs danew_dtau;

        /// \brief Contact solver derivatives.
        ContactSolverDerivatives contact_solver_derivatives;

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
        /// \brief Vector of ContactPatchDerivativeRequest.
        std::vector<ContactPatchDerivativeRequest> patch_derivative_requests;

        /// \brief Vector of ContactPatchDerivative.
        std::vector<ContactPatchDerivative> patch_derivatives;
#endif

        SIMPLEX_PROTECTED
        /// \brief Memory containing MinvJT operator;
        RowMatrixXs MinvJT_;

        /// \brief Memory containing partial derivatives of the contact point velocities wrt q, v and
        /// tau.
        MatrixXs dGlamg_dtheta_;

        /// \brief Memory containing partial derivatives of the contact point velocities wrt v.
        MatrixXs dGlamg_dvnew_;

        /// \brief Partial derivatives wrt q of contact point velocity of shape 1.
        Matrix6Xs dsigma1_dq_;

        /// \brief Partial derivatives wrt q of contact point velocity of shape 2.
        Matrix6Xs dsigma2_dq_;

        /// \brief Partial derivatives wrt v of contact point velocity of shape 1.
        Matrix6Xs dsigma1_dv_;

        /// \brief Partial derivatives wrt v of contact point velocity of shape 2.
        Matrix6Xs dsigma2_dv_;

        /// \brief Timer used for evaluating timings of stepDerivatives.
        coal::Timer timer_;

        /// \brief Timings for the call to the contact solver.
        coal::CPUTimes timings_;

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
        /// \brief Memory containing the Primal Collision detection Corrective terms.
        /// This correspongs to dJc*v/dq, with v considered fixed.
        /// This term takes into account all contact points of the system.
        MatrixXs primal_collision_correction_;

        /// \brief Memory containing the Dual Collision detection Corrective terms.
        /// This correspongs to Minv * dJcT*lam/dq, with lam considered fixed.
        /// This term takes into account all contact points of the system.
        MatrixXs dual_collision_correction_;

        /// \brief Accumulator for dJcT*lam/dq for all contact points.
        MatrixXs dJcTlam_dq_;

        /// \brief Temporary holder for the contact jacobian of a single contact point.
        Matrix6Xs Jc_;

        /// \brief Temporary holder for dJc*v/dq for a single contact point.
        Matrix6Xs dJcv_dq_;

        /// \brief Temporary holder for a jacobian related to the first geom of a collision pair.
        Matrix6Xs J1_;

        /// \brief Temporary holder for a jacobian related to the second geom of a collision pair.
        Matrix6Xs J2_;

        /// \brief Temporary holder for a doMc/dq, for a single contact point.
        Matrix6Xs doMc_dq_;

        /// \brief Temporary for the dual ad swap operator.
        Matrix6Xs dual_swap_doMc_dq_;

        /// \brief Vector of contact patch derivative functors.
        std::vector<ComputeContactPatchDerivative> patch_derivative_functors_;
#endif

        SIMPLEX_PUBLIC
        ///
        /// \brief Default constructor
        SimulatorDerivativesTpl(const SimulatorX & simulator)
        : measure_timings(false)
        , contact_solver_derivatives(simulator.workspace.getConstraintProblemHandle())
#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
        , patch_derivative_requests(simulator.geom_model().collisionPairs.size())
        , patch_derivatives(simulator.geom_model().collisionPairs.size())
#endif
        , timer_(false)
        {
            allocate(simulator);
        }

        ///
        /// \brief Computes Jacobian of the step function wrt q, v and tau.
        /// note: step must be called on the simulator before calling this function.
        template<typename ConfigVectorType, typename VelocityVectorType, typename TorqueVectorType>
        void stepDerivatives(
            SimulatorX & simulator,
            const Eigen::MatrixBase<ConfigVectorType> & q,
            const Eigen::MatrixBase<VelocityVectorType> & v,
            const Eigen::MatrixBase<TorqueVectorType> & tau,
            const Scalar dt);

        /// \brief Get timings of the call to the derivatives of step.
        coal::CPUTimes getSimulatorDerivativesCPUTimes() const
        {
            return timings_;
        }

        SIMPLEX_PROTECTED
        ///
        /// \brief Allocates memory based on the simulator.
        void allocate(const SimulatorX & simulator);

        ///
        /// \brief Getter for the partial derivatives of the contact point velocities.
        Eigen::Block<MatrixXs> dGlamg_dtheta() // TODO replace by an eigen map on a vectorXS
        {
            const ContactIndex idx = (ContactIndex)(3 * nc);
            return dGlamg_dtheta_.topRows(idx);
        }

        const Eigen::Block<const MatrixXs> dGlamg_dtheta() const // TODO replace by an eigen map on a vectorXS
        {
            const ContactIndex idx = (ContactIndex)(3 * nc);
            return dGlamg_dtheta_.topRows(idx);
        }

        ///
        /// \brief Getter for the MinvJT operator.
        Eigen::Block<RowMatrixXs, Eigen::Dynamic, Eigen::Dynamic, false> MinvJT() // TODO replace by an eigen map on a vectorXS
        {
            const ContactIndex idx = (ContactIndex)(3 * nc);
            return MinvJT_.leftCols(idx);
        }

        const Eigen::Block<const RowMatrixXs, Eigen::Dynamic, Eigen::Dynamic, false>
        MinvJT() const // TODO replace by an eigen map on a vectorXS
        {
            const ContactIndex idx = (ContactIndex)(3 * nc);
            return MinvJT_.leftCols(idx);
        }

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
        ///
        /// \brief Getter for the primal collision correction terms.
        Eigen::Block<MatrixXs> primal_collision_correction() // TODO replace by an eigen map on a vectorXS
        {
            const ContactIndex idx = (ContactIndex)(3 * nc);
            return primal_collision_correction_.topRows(idx);
        }

        ///
        /// \brief Getter for the primal collision correction terms.
        Eigen::Block<const MatrixXs> primal_collision_correction() const // TODO replace by an eigen map on a vectorXS
        {
            const ContactIndex idx = (ContactIndex)(3 * nc);
            return primal_collision_correction_.topRows(idx);
        }

        ///
        /// \brief Getter for the dual collision correction terms.
        MatrixXs & dual_collision_correction() // TODO replace by an eigen map on a vectorXS
        {
            return dual_collision_correction_;
        }

        ///
        /// \brief Getter for the dual collision correction terms.
        const MatrixXs & dual_collision_correction() const // TODO replace by an eigen map on a vectorXS
        {
            return dual_collision_correction_;
        }

        ///
        /// \brief Compute the primal/dual collision correction terms.
        template<typename VelocityVectorType>
        void computePrimalDualCollisionCorrection(const SimulatorX & simulator, const Eigen::MatrixBase<VelocityVectorType> & v);

#endif
    };

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    ///
    /// \brief Constructs the swap of the dual of the small adjoint.
    /// This operator encodes the operation ad(m)^T acting on f, where m is a spatial motion and f a
    /// spatial force. The dualSmallAdSwap has the following property:
    /// Let mb = ::pinocchio::Motion(b) and fa = ::pinocchio::Force(a), then:
    ///   coswap = dualSmallAdSwap(fa) and we have
    ///     coswap @ b - (mb.action).transpose() @ a = [0, 0, 0, 0, 0, 0]
    template<typename Scalar, int Options, typename Matrix6Like>
    void dualSmallAdSwap(const ::pinocchio::ForceTpl<Scalar, Options> & wrench, const Eigen::MatrixBase<Matrix6Like> & res)
    {
        EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Matrix6Like, 6, 6);
        Matrix6Like & res_ = const_cast<Matrix6Like &>(res.derived());
        res_.setZero();
        res_.template topRightCorner<3, 3>() = ::pinocchio::skew(wrench.linear());
        res_.template bottomLeftCorner<3, 3>() = ::pinocchio::skew(wrench.linear());
        res_.template bottomRightCorner<3, 3>() = ::pinocchio::skew(wrench.angular());
    }
#endif

} // namespace simplex

/* --- Details -------------------------------------------------------------- */
#include "simplex/core/simulator-derivatives.hxx"

#if SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION
    #include "simplex/core/simulator-derivatives.txx"
    #include "simplex/pinocchio_template_instantiation/joint-model-inst.hpp"
    #include "simplex/pinocchio_template_instantiation/aba-derivatives-inst.hpp"
#endif

#endif // ifndef __simplex_core_simulator_derivatives_hpp__
