#include "simplex/core/contact-frame.hpp"

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

using namespace simplex;
using namespace pinocchio;
using Vector3d = Eigen::Vector3d;

BOOST_AUTO_TEST_SUITE(BOOST_TEST_MODULE)

BOOST_AUTO_TEST_CASE(contact_frame)
{
    const Vector3d normal = Vector3d::Random().normalized();
    const Vector3d position = Vector3d::Random();
    ::pinocchio::SE3 M;
    PlacementFromNormalAndPosition::calc(normal, position, M);

    BOOST_CHECK((M.translation().isApprox(position)));
    BOOST_CHECK((M.rotation() * (M.rotation().transpose())).isIdentity());
    BOOST_CHECK((M.rotation().determinant() - 1) <= 1e-12);
}

BOOST_AUTO_TEST_CASE(contact_frame_derivative)
{
    const Vector3d normal = Vector3d::Random().normalized();
    const Vector3d position = Vector3d::Random();
    ::pinocchio::SE3 M;
    PlacementFromNormalAndPosition::calc(normal, position, M);
    using Matrix63 = PlacementFromNormalAndPosition::Matrix63s;
    Matrix63 dM_dn;
    Matrix63 dM_dp;
    PlacementFromNormalAndPosition::calcDiff(M, dM_dn, dM_dp);

    // Check dM_dn
    Vector3d ei(0, 0, 0);
    static constexpr double eps = 1e-6;
    ::pinocchio::SE3 Mplus;
    ::pinocchio::SE3 Mminus;
    Matrix63 dM_dn_fd;
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        ei(i) = eps;
        const Vector3d v = normal + ei;
        const Vector3d dn = v - normal * (normal.dot(v));
        PlacementFromNormalAndPosition::calc(normal + dn, position, Mplus);
        PlacementFromNormalAndPosition::calc(normal - dn, position, Mminus);
        dM_dn_fd.col(i) = (::pinocchio::log6(Mminus.inverse() * Mplus).toVector()) / (2 * eps);
        ei(i) = 0;
    }
    BOOST_CHECK(dM_dn.isApprox(dM_dn_fd, 1e-6));

    // Check dM_dp
    Matrix63 dM_dp_fd;
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        ei(i) = eps;
        PlacementFromNormalAndPosition::calc(normal, position + ei, Mplus);
        PlacementFromNormalAndPosition::calc(normal, position - ei, Mminus);
        dM_dp_fd.col(i) = ::pinocchio::log6(Mminus.inverse() * Mplus).toVector() / (2 * eps);
        ei(i) = 0;
    }
    BOOST_CHECK(dM_dp.isApprox(dM_dp_fd, 1e-6));
}

BOOST_AUTO_TEST_SUITE_END()
