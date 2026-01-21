#include "simplex/bindings/python/fwd.hpp"
#include "simplex/config.hpp"
#include "simplex/fwd.hpp"

#include <eigenpy/eigenpy.hpp>

namespace bp = boost::python;
using namespace simplex::python;

BOOST_PYTHON_MODULE(SIMPLEX_PYTHON_MODULE_NAME)
{
    bp::docstring_options module_docstring_options(true, true, false);

    bp::scope().attr("__version__") = bp::str(SIMPLEX_VERSION);
    bp::scope().attr("__raw_version__") = bp::str(SIMPLEX_VERSION);

    eigenpy::enableEigenPy();
    using Matrix63s = Eigen::Matrix<simplex::context::Scalar, 6, 3, simplex::context::Options>;
    eigenpy::enableEigenPySpecific<Matrix63s>();

    // Enable warnings
    bp::import("warnings");

    // Dependencies
    bp::import("coal");
    bp::import("pinocchio");

    exposeContactFrame();
    exposeConstraintsProblemDerivatives();
    exposeClarabelSolver();
    exposeSimulatorX();
    exposeSimulatorDerivatives();
    exposeContactSolverDerivatives();
}
