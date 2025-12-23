#ifndef __simplex_core_constraints_problem_txx__
#define __simplex_core_constraints_problem_txx__

#include "simplex/core/constraints-problem.hpp"

namespace simplex
{

    extern template struct ConstraintsProblemTpl<context::Scalar, context::Options, ::pinocchio::JointCollectionDefaultTpl>;

} // namespace simplex

#endif // __simplex_core_constraints_problem_txx__