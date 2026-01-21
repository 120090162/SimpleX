#include "simplex/bindings/python/core/simulator-x.hpp"

namespace simplex
{
    namespace python
    {

        void exposeSimulatorX()
        {
            SimulatorXPythonVisitor<SimulatorX>::expose();
        }

    } // namespace python
} // namespace simplex
