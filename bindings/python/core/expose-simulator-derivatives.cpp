#include "simplex/bindings/python/core/simulator-derivatives.hpp"

namespace simplex
{
    namespace python
    {

        void exposeSimulatorDerivatives()
        {
            SimulatorDerivativesPythonVisitor<SimulatorDerivatives>::expose();
        }

    } // namespace python
} // namespace simplex
