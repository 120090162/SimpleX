#ifndef __simplex_core_simulator_derivatives_txx__
#define __simplex_core_simulator_derivatives_txx__

#include "simplex/core/simulator-derivatives.hpp"

namespace simplex
{
    extern template struct SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

    extern template void
    SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::stepDerivatives(
        SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    extern template void
    SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::computeDualCorrection(
        const SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
        const Eigen::MatrixBase<context::VectorXs> &);

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    extern template void SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::
        computePrimalDualCollisionCorrection(
            const SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
            const Eigen::MatrixBase<context::VectorXs> &);
#endif

} // namespace simplex

#endif // __simplex_core_simulator_derivatives_txx__
