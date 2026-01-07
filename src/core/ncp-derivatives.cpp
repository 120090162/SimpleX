#include "simplex/core/ncp-derivatives.hpp"

namespace simplex
{

    template struct ContactSolverDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

    template void ContactSolverDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>::jvp(
        const Eigen::MatrixBase<Eigen::Ref<const context::MatrixXs>> &);

} // namespace simplex