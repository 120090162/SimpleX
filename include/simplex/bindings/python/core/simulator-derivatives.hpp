#ifndef __simplex_python_core_simulator_derivatives_hpp__
#define __simplex_python_core_simulator_derivatives_hpp__

#include "simplex/core/simulator-derivatives.hpp"
#include "simplex/bindings/python/fwd.hpp"

#include <pinocchio/bindings/python/utils/copyable.hpp>

namespace simplex
{
    namespace python
    {
        namespace bp = boost::python;

        template<typename SimulatorDerivatives>
        struct SimulatorDerivativesPythonVisitor
        : public boost::python::def_visitor<SimulatorDerivativesPythonVisitor<SimulatorDerivatives>>
        {
            using Self = SimulatorDerivatives;
            using Scalar = typename Self::Scalar;
            using VectorXs = typename Self::VectorXs;
            using MatrixXs = typename Self::MatrixXs;

            using ConstraintsProblemDerivatives = typename Self::ConstraintsProblemDerivatives;
            using SimulatorX = typename Self::SimulatorX;

            template<class PyClass>
            void visit(PyClass & cl) const
            {
                cl.def(bp::init<SimulatorX>(bp::args("self", "simulator"), "Constructor"))
                    .def_readwrite("measure_timings", &Self::measure_timings, "Measure timings of the `step` method.")
                    .def_readonly("dvnew_dq", &Self::dvnew_dq, "Jacobian of the velocity wrt q.")
                    .def_readonly("dvnew_dv", &Self::dvnew_dv, "Jacobian of the velocity wrt v.")
                    .def_readonly("dvnew_dtau", &Self::dvnew_dtau, "Jacobian of the velocity wrt tau.")
                    .def_readonly("contact_solver_derivatives", &Self::contact_solver_derivatives, "Contact solver derivatives.")

                    .def(
                        "stepDerivatives",
                        +[](Self & self, SimulatorX & simulator, const VectorXs & q, const VectorXs & v, const VectorXs & tau, Scalar dt) {
                            self.template stepDerivatives<>(simulator, q, v, tau, dt);
                        },
                        (bp::arg("self"), bp::arg("simulator"), bp::arg("q"), bp::arg("v"), bp::arg("tau"), bp::arg("dt")),
                        "Compute the Jacobian of the step function wrt q, v and tau.")

                    .def(
                        "getSimulatorDerivativesCPUTimes", &Self::getSimulatorDerivativesCPUTimes, bp::args("self"),
                        "Get timings of the call to the contact solver in the last call to the `step` method. "
                        "These timings can be 0 if no contacts occured.");
            }

            static void expose()
            {
                bp::class_<SimulatorDerivatives>("SimulatorDerivatives", "Class computing derivatives of the simulator.", bp::no_init)
                    .def(SimulatorDerivativesPythonVisitor<SimulatorDerivatives>())
                    .def(::pinocchio::python::CopyableVisitor<SimulatorDerivatives>());
            }
        };

    } // namespace python
} // namespace simplex

#endif // ifndef __simplex_python_core_simulator_derivatives_hpp__
