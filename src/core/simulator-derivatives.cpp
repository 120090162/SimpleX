#include "simplex/core/simulator-derivatives.hpp"

namespace simplex
{
    template struct SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

    template void SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::stepDerivatives(
        SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    template void SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::computeDualCorrection(
        const SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
        const Eigen::MatrixBase<context::VectorXs> &);

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    template void SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::
        computePrimalDualCollisionCorrection(
            const SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
            const Eigen::MatrixBase<context::VectorXs> &);
#endif

} // namespace simplex
