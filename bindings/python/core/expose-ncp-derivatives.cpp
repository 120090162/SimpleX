#include "simplex/bindings/python/core/ncp-derivatives.hpp"

namespace simplex
{
    namespace python
    {

        void exposeContactSolverDerivatives()
        {
            using ImplicitGradientSystemSolver = ContactSolverDerivatives::ImplicitGradientSystemSolver;
            bp::enum_<ImplicitGradientSystemSolver>("ImplicitGradientSystemSolver")
                .value("HOUSEHOLDER_QR", ImplicitGradientSystemSolver::HOUSEHOLDER_QR)
                .value("COD", ImplicitGradientSystemSolver::COD)
                .export_values();

            ContactSolverDerivativesPythonVisitor<ContactSolverDerivatives>::expose();
        }

    } // namespace python
} // namespace simplex
