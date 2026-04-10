#include <iostream>
#include "contactbench/contact-problem.hpp"
#include "contactbench/solvers.hpp"
#include <vector>
#include <Eigen/Eigenvalues>
#include <stdexcept>

namespace cb = contactbench;
using T = double;
CONTACTBENCH_EIGEN_TYPEDEFS(T);

bool solve_contact_bench_problem()
{
    int nc = 4;
    MatrixXs A = MatrixXs::Zero(12, 12);
    A << 1.70260293, -0.47810656, -0.81113264, 0.87470029, -0.47808302,
        -0.81160892, 1.70263763, 0.4774236, 0.81159737, 0.87473499, 0.47744715,
        0.81112109, -0.47810656, 2.40658732, -1.16458464, 0.46986558, 2.40900655,
        1.15727014, -0.47813832, 1.3144438, -1.16816273, 0.46983382, 1.31686302,
        1.15369205, -0.81113264, -1.16458464, 2.07287338, -0.80996834,
        -1.16653817, 0.21909843, -0.81113907, -1.16749827, 0.58960997,
        -0.80997476, -1.1694518, -1.26416498, 0.87470029, 0.46986558, -0.80996834,
        1.69128967, 0.46984242, -0.80943495, 0.87467992, -0.46921415, 0.80941869,
        1.6912693, -0.4692373, 0.80995208, -0.47808302, 2.40900655, -1.16653817,
        0.46984242, 2.41142987, 1.15920541, -0.47811477, 1.31691684, -1.17011623,
        0.46981067, 1.31934017, 1.15562734, -0.81160892, 1.15727014, 0.21909843,
        -0.80943495, 1.15920541, 2.0556279, -0.81161538, 1.15321151, -1.26423198,
        -0.80944141, 1.15514678, 0.57229749, 1.70263763, -0.47813832, -0.81113907,
        0.87467992, -0.47811477, -0.81161538, 1.70267234, 0.47745531, 0.81160383,
        0.87471463, 0.47747886, 0.81112752, 0.4774236, 1.3144438, -1.16749827,
        -0.46921415, 1.31691684, 1.15321151, 0.47745531, 2.40507282, -1.16409878,
        -0.46918244, 2.40754587, 1.156611, 0.81159737, -1.16816273, 0.58960997,
        0.80941869, -1.17011623, -1.26423198, 0.81160383, -1.16409878, 2.07291549,
        0.80942515, -1.16605229, 0.21907354, 0.87473499, 0.46983382, -0.80997476,
        1.6912693, 0.46981067, -0.80944141, 0.87471463, -0.46918244, 0.80942515,
        1.69124894, -0.46920559, 0.80995851, 0.47744715, 1.31686302, -1.1694518,
        -0.4692373, 1.31934017, 1.15514678, 0.47747886, 2.40754587, -1.16605229,
        -0.46920559, 2.41002301, 1.15854629, 0.81112109, 1.15369205, -1.26416498,
        0.80995208, 1.15562734, 0.57229749, 0.81112752, 1.156611, 0.21907354,
        0.80995851, 1.15854629, 2.05553602;
    VectorXs b = VectorXs::Zero(12);
    b << 0.00554428, 0.00582766, -0.00701578, 0.00820943, 0.00582485, -0.0096867,
        0.00554418, 0.00275529, -0.00877465, 0.00820934, 0.00275248, -0.01144557;
    std::vector<T> mus = {0.9, 0.9, 0.9, 0.9};
    cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
    cb::RaisimSolver<T> solver;
    solver.setProblem(prob);
    VectorXs x0 = VectorXs::Zero(12);
    x0 << -9.41930386e-05, -1.19316259e-04, 1.68906060e-04, -1.19463177e-04,
        -6.29330529e-04, 7.11743091e-04, -6.79642646e-03, 4.00495120e-03,
        8.76518412e-03, -4.56331805e-03, -2.79577045e-03, 5.94628258e-03;
    VectorXs v0 = A * x0 + b;
    prob.Del_->evaluateDel();
    solver.computeGinv(prob.Del_->G_);
    solver.computeC(prob.Del_->G_, prob.g_, x0);
    Vector3s lam_v0, c_j, c_j2, lam_star;
    Vector2s lam_star_t_cor, v_star_t;
    Matrix3s G_j, Ginv_j;
    int maxIter = 10000;
    double th = 1e-8;
    double beta1 = 1e-2;
    double beta2 = 0.5;
    double beta3 = 1.3;
    bool test_passed = true;

    for (int i = 0; i < nc; i++)
    {
        c_j = solver.getC(i);
        G_j = prob.Del_->G_.block<3, 3>(3 * i, 3 * i);
        Ginv_j = solver.getGinv(i);
        solver.computeLamV0(Ginv_j, c_j, lam_v0);
        c_j2 = c_j + G_j * lam_v0;
        if (c_j2.norm() >= 1e-6)
        {
            std::cerr << "Test failed at c_j2.norm() < 1e-6" << std::endl;
            test_passed = false;
            break;
        }
        solver.bisectionStep(G_j, Ginv_j, c_j, mus[CAST_UL(i)], lam_v0, lam_star,
                             maxIter, th, beta1, beta2, beta3);
        if (!prob.contact_constraints_[CAST_UL(i)].isOnBorder(lam_star, 1e-5))
        {
            std::cerr << "Test failed at contact constraint check" << std::endl;
            test_passed = false;
            break;
        }
        c_j2 = c_j + G_j * lam_star;
        if (std::abs(c_j2(2)) >= 1e-5)
        {
            std::cerr << "Test failed at c_j2(2) < 1e-5" << std::endl;
            test_passed = false;
            break;
        }
        v_star_t = c_j2.head<2>();
        v_star_t.normalize();
        lam_star_t_cor = -lam_star.head<2>();
        lam_star_t_cor += (-(mus[CAST_UL(i)] * mus[CAST_UL(i)] * lam_star(2) / G_j(2, 2)) * G_j.row(2).head<2>());
        lam_star_t_cor.normalize();
    }

    if (!test_passed)
        return false;

    cb::ContactSolverSettings<T> settings;
    settings.max_iter_ = maxIter;
    settings.th_stop_ = 1e-12;
    settings.rel_th_stop_ = 1e-12;
    settings.statistics_ = true;
    solver.solve(prob, x0, settings, 0., 1., 0.1, 1e-2, .5, 1.3, 0.99, 1e-6);
    VectorXs lam = solver.getSolution();
    VectorXs v = A * lam + b;
    double comp = prob.computeContactComplementarity(lam);

    VectorXs lam_t_cor = VectorXs::Zero(2 * nc);
    VectorXs lam_t = VectorXs::Zero(2 * nc);
    VectorXs v_t = VectorXs::Zero(2 * nc);
    for (int i = 0; i < nc; i++)
    {
        lam_t_cor.segment<2>(2 * i) = A.row(3 * i + 2).segment<2>(3 * i);
        lam_t_cor.segment<2>(2 * i) *= (-(mus[CAST_UL(i)] * mus[CAST_UL(i)]) * lam(3 * i + 2) / A(3 * i + 2, 3 * i + 2));
        lam_t_cor.segment<2>(2 * i) += -lam.segment<2>(3 * i);
        lam_t_cor.segment<2>(2 * i).normalize();
        lam_t.segment<2>(2 * i) = -lam.segment<2>(3 * i);
        lam_t.segment<2>(2 * i).normalize();
        v_t.segment<2>(2 * i) = v.segment<2>(3 * i);
        v_t.segment<2>(2 * i).normalize();
    }

    if (!v_t.tail<2>().isApprox(lam_t_cor.tail<2>(), 1e-3))
    {
        std::cerr << "Test failed at v_t.tail<2>().isApprox(lam_t_cor.tail<2>(), 1e-3)" << std::endl;
        test_passed = false;
    }

    return test_passed;
}

int main()
{
    try
    {
        bool result = solve_contact_bench_problem();
        if (result)
        {
            std::cout << "All tests passed." << std::endl;
        }
        else
        {
            std::cerr << "Some tests failed." << std::endl;
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}