//
// Copyright (c) 2022-2024 INRIA
//

#ifndef __simple_simulator_txx__
#define __simple_simulator_txx__

#include "simple/core/simulator.hpp"

namespace simple
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

} // namespace simple

#endif // __simple_simulator_txx__
