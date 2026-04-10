#include "Eigen/Dense"
#include <iostream>

//************************
// main function
int main(int argc, const char **argv)
{
    // 创建两个3x3矩阵初始化为特定值
    Eigen::Matrix3d matrix_1;
    matrix_1 << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    Eigen::Matrix3d matrix_2;
    matrix_2 << 9, 8, 7,
        6, 5, 4,
        3, 2, 1;

    std::cout << "Matrix 1:\n"
              << matrix_1 << std::endl;
    std::cout << "Matrix 2:\n"
              << matrix_2 << std::endl;

    // 矩阵相加
    Eigen::Matrix3d sum = matrix_1 + matrix_2;
    std::cout << "Sum of Matrix 1 and Matrix 2:\n"
              << sum << std::endl;

    // 矩阵相乘
    Eigen::Matrix3d product = matrix_1 * matrix_2;
    std::cout << "Product of Matrix 1 and Matrix 2:\n"
              << product << std::endl;

    // 使用Eigen求解线性方程组 Ax = b 形式
    Eigen::Vector3d b(12, 12, 12); // 右侧向量
    Eigen::Vector3d x;

    // 求解matrix_1 * x = b
    x = matrix_1.colPivHouseholderQr().solve(b);
    std::cout << "Solution of Ax = b using colPivHouseholderQr():\n"
              << x << std::endl;

    return 0;
}