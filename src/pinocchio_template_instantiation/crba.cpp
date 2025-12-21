#include "simplex/fwd.hpp"
#include <pinocchio/algorithm/crba.hpp>

template PINOCCHIO_EXPLICIT_INSTANTIATION_DEFINITION_DLLAPI const ::simplex::context::MatrixXs & ::pinocchio::
	crba<::simplex::context::Scalar, ::simplex::context::Options, ::pinocchio::JointCollectionDefaultTpl, ::simplex::context::VectorXs>(
		const ::simplex::context::Model &, ::simplex::context::Data &, const Eigen::MatrixBase<::simplex::context::VectorXs> &, const Convention convention);