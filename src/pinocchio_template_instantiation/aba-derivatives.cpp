#include "simplex/fwd.hpp"
#include <pinocchio/algorithm/aba-derivatives.hpp>

template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI void ::pinocchio::computeABADerivatives<
	::simplex::context::Scalar,
	::simplex::context::Options,
	::pinocchio::JointCollectionDefaultTpl,
	::simplex::context::VectorXs,
	::simplex::context::VectorXs,
	::simplex::context::VectorXs,
	::simplex::context::Force,
	Eigen::aligned_allocator<::simplex::context::Force>,
	Eigen::Ref<::simplex::context::RowMatrixXs>,
	Eigen::Ref<::simplex::context::RowMatrixXs>,
	Eigen::Ref<::simplex::context::RowMatrixXs>>(
	const ::simplex::context::Model &,
	::simplex::context::Data &,
	const Eigen::MatrixBase<::simplex::context::VectorXs> &,
	const Eigen::MatrixBase<::simplex::context::VectorXs> &,
	const Eigen::MatrixBase<::simplex::context::VectorXs> &,
	const std::vector<::simplex::context::Force, Eigen::aligned_allocator<::simplex::context::Force>> &,
	const Eigen::MatrixBase<Eigen::Ref<::simplex::context::RowMatrixXs>> &,
	const Eigen::MatrixBase<Eigen::Ref<::simplex::context::RowMatrixXs>> &,
	const Eigen::MatrixBase<Eigen::Ref<::simplex::context::RowMatrixXs>> &);