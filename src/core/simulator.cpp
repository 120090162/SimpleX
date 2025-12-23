#include "simplex/core/simulator.hpp"

namespace simplex
{
    template struct SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>;

    template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    template void SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::pinocchio::ADMMContactSolverTpl>(const Eigen::MatrixBase<context::VectorXs> &, const Scalar);

    template void SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::pinocchio::PGSContactSolverTpl>(const Eigen::MatrixBase<context::VectorXs> &, const Scalar);

} // namespace simplex
