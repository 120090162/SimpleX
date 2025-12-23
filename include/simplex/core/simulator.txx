#ifndef __simplex_simulator_txx__
#define __simplex_simulator_txx__

#include "simplex/core/simulator.hpp"

namespace simplex
{
    extern template struct SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>;

    extern template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    extern template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::ADMMContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    extern template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        context::Scalar);

    extern template void
    SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::step<::pinocchio::PGSContactSolverTpl>(
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const Eigen::MatrixBase<context::VectorXs> &,
        const ::pinocchio::container::aligned_vector<::pinocchio::ForceTpl<context::Scalar>> &,
        context::Scalar);

    extern template void SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::pinocchio::ADMMContactSolverTpl>(const Eigen::MatrixBase<context::VectorXs> &, const Scalar);

    extern template void SimulatorTpl<context::Scalar, context::Options, pinocchio::JointCollectionDefaultTpl>::resolveConstraints<
        ::pinocchio::PGSContactSolverTpl>(const Eigen::MatrixBase<context::VectorXs> &, const Scalar);

} // namespace simplex

#endif // __simplex_simulator_txx__
