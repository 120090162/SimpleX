#ifndef __simplex__test_utils_hpp__
#define __simplex__test_utils_hpp__

#define EIGEN_VECTOR_IS_APPROX(Va, Vb, precision)                                                                                          \
    BOOST_CHECK_MESSAGE(                                                                                                                   \
        ((Va) - (Vb)).isZero(precision), "check " #Va ".isApprox(" #Vb ") failed at precision "                                            \
                                             << precision << ". (" #Va " - " #Vb ").norm() = " << ((Va) - (Vb)).norm() << " [\n"           \
                                             << (Va).transpose() << "\n!=\n"                                                               \
                                             << (Vb).transpose() << "\n]")

#define INDEX_EQUALITY_CHECK(i1, i2) BOOST_CHECK_MESSAGE(i1 == i2, "check " #i1 "==" #i2 " failed. [" << i1 << " != " << i2 << "]")
#define INDEX_INEQUALITY_CHECK(i1, i2) BOOST_CHECK_MESSAGE(i1 <= i2, "check " #i1 "==" #i2 " failed. [" << i1 << " != " << i2 << "]")

#define REAL_IS_APPROX(a, b, precision)                                                                                                    \
    BOOST_CHECK_MESSAGE(                                                                                                                   \
        std::abs((a) - (b)) < precision,                                                                                                   \
        "check std::abs(" #a " - " #b ") = " << std::abs((a) - (b)) << " < " << precision << " failed. [" << a << " != " << b << "]")

#define FMT_HEADER_ONLY
#include <fmt/format.h>

inline std::string findTestResource(const std::string & simple_test_resource_path)
{
    std::cout << "Finding test resource: " << simple_test_resource_path << std::endl;
    std::cout << "SIMPLEX_TEST_DIR: " << SIMPLEX_TEST_DIR << std::endl;
    if (simple_test_resource_path.substr(0, 8) != "SIMPLEX/")
    {
        throw std::runtime_error(fmt::format("Resource path '{}' must start with 'SIMPLEX/' ", simple_test_resource_path));
    }
    // N.B. SIMPLEX_TEST_DIR set in top-level CMakeLists.txt
    return std::string(SIMPLEX_TEST_DIR) + simple_test_resource_path.substr(7, simple_test_resource_path.size() - 7);
}

#endif // __simplex__test_utils_hpp__
