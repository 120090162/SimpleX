#include "simplex/core/contact-frame.hpp"
#include "simplex/bindings/python/fwd.hpp"

#include <boost/python/tuple.hpp>

namespace bp = boost::python;

namespace simplex
{
    namespace python
    {
        using SE3 = ::simplex::PlacementFromNormalAndPosition::SE3;
        using Vector3s = ::simplex::PlacementFromNormalAndPosition::Vector3s;
        using Matrix63s = ::simplex::PlacementFromNormalAndPosition::Matrix63s;

        SE3 placementFromNormalAndPosition(const Vector3s & normal, const Vector3s & position)
        {
            SE3 M;
            ::simplex::PlacementFromNormalAndPosition::calc(normal, position, M);
            return M;
        }

        bp::tuple placementFromNormalAndPositionDerivative(const SE3 & M)
        {
            Matrix63s dM_dnormal;
            Matrix63s dM_dposition;
            ::simplex::PlacementFromNormalAndPosition::calcDiff(M, dM_dnormal, dM_dposition);
            return bp::make_tuple(dM_dnormal, dM_dposition);
        }

        void exposeContactFrame()
        {
            bp::def(
                "placementFromNormalAndPosition", placementFromNormalAndPosition, bp::args("normal", "position"),
                "Returns a placement such that `normal` is the z-axis of the placement's rotation and "
                "`position` is the translation of the placement.\n"
                "Parameters:\n"
                "\tnormal: z-axis of the placement's rotation.\n"
                "\tposition: translation part of the placement.\n\n"
                "Returns: M, the placement.");

            bp::def(
                "placementFromNormalAndPositionDerivative", placementFromNormalAndPositionDerivative, bp::args("M"),
                "Returns the derivatives of a placement w.r.t both the normal and position than generated "
                "it. The normal is the z-axis of the placement's rotation and the position is the "
                "translation of the part of the placement."
                "Parameters:\n"
                "\tM: a placement.\n\n"
                "Returns: a tuple (dM_dnormal, dM_dposition), both are 6x3 matrices.");
        }

    } // namespace python
} // namespace simplex
