#include "simplex/bindings/python/solver/clarabel-solver.hpp"

namespace simplex
{
    namespace python
    {

        void exposeClarabelSolver()
        {
#ifdef SIMPLEX_WITH_CLARABEL_SUPPORT
            ClarabelConstraintSolverPythonVisitor<::simplex::ClarabelContactSolverTpl<context::Scalar, context::Options>>::expose();
#endif
        }

    } // namespace python
} // namespace simplex
