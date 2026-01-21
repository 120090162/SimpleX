#ifndef __simplex_python_fwd_hpp__
#define __simplex_python_fwd_hpp__

#include <eigenpy/eigenpy.hpp>

namespace simplex
{
    namespace python
    {
        void exposeContactFrame();
        void exposeConstraintsProblemDerivatives();
        void exposeClarabelSolver();
        void exposeSimulatorX();
        void exposeSimulatorDerivatives();
        void exposeContactSolverDerivatives();
    } // namespace python
} // namespace simplex

#endif // #ifndef __simplex_python_fwd_hpp__
