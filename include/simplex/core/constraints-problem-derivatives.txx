#ifndef __simplex_core_constraints_problem_derivatives_txx__
#define __simplex_core_constraints_problem_derivatives_txx__

#include "simplex/core/constraints-problem-derivatives.hpp"

namespace simplex
{

    extern template struct ConstraintsProblemDerivativesTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

} // namespace simplex

#endif // __simplex_core_constraints_problem_derivatives_txx__