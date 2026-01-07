#include "simplex/core/simulator-derivatives.hpp"
#include "simplex/pinocchio_template_instantiation/joint-model-inst.hpp"
#include "simplex/pinocchio_template_instantiation/aba-derivatives-inst.hpp"

namespace simplex
{
    template struct SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

    template void SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::stepDerivatives(
        SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

#ifndef SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS
    template void SimulatorDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::
        computePrimalDualCollisionCorrection(
            const SimulatorXTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl> &,
            const Eigen::MatrixBase<context::VectorXs> &);
#endif

} // namespace simplex
