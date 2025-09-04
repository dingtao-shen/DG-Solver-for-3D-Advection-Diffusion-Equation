#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "FiniteElement/BasisFunctions.hpp"

using std::cerr;
using std::cout;
using std::endl;

namespace {

int failures = 0;

void require(bool condition, const std::string &message) {
    if (!condition) {
        ++failures;
        cerr << "[FAIL] " << message << endl;
    }
}

bool approxEqual(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b)));
}

bool approxEqual(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B, double tol = 1e-12) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) return false;
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            if (!approxEqual(A(i, j), B(i, j), tol)) return false;
        }
    }
    return true;
}

void testEquiNodes() {
    // 1D, order 0
    {
        Eigen::MatrixXd nodes = EquiNodes(1, 0);
        require(nodes.rows() == 1 && nodes.cols() == 1, "EquiNodes(1,0) size");
        require(approxEqual(nodes(0, 0), 0.0), "EquiNodes(1,0) value");
    }
    // 2D, order 0
    {
        Eigen::MatrixXd nodes = EquiNodes(2, 0);
        require(nodes.rows() == 2 && nodes.cols() == 1, "EquiNodes(2,0) size");
        require(approxEqual(nodes, Eigen::MatrixXd::Zero(2, 1)), "EquiNodes(2,0) values");
    }
    // 3D, order 0
    {
        Eigen::MatrixXd nodes = EquiNodes(3, 0);
        require(nodes.rows() == 3 && nodes.cols() == 1, "EquiNodes(3,0) size");
        require(approxEqual(nodes, Eigen::MatrixXd::Zero(3, 1)), "EquiNodes(3,0) values");
    }
    // 2D, order 1 -> 3 nodes
    {
        Eigen::MatrixXd nodes = EquiNodes(2, 1);
        require(nodes.rows() == 2 && nodes.cols() == 3, "EquiNodes(2,1) dof");
        // Expect the standard simplex vertices in some order
        // All nodes should satisfy x >= 0, y >= 0, x + y <= 1
        for (int i = 0; i < nodes.cols(); ++i) {
            require(nodes(0, i) >= -1e-14, "EquiNodes(2,1) x >= 0");
            require(nodes(1, i) >= -1e-14, "EquiNodes(2,1) y >= 0");
            require(nodes(0, i) + nodes(1, i) <= 1.0 + 1e-14, "EquiNodes(2,1) x+y <= 1");
        }
    }
}

void testBarycentricCartesian() {
    // 2D
    {
        Eigen::MatrixXd xy(2, 3);
        xy << 0.2, 1.0, 0.0,
              0.3, 0.0, 1.0;
        Eigen::MatrixXd L = Barycentric(2, xy);
        // L0 = 1 - x - y, L1 = x, L2 = y
        require(approxEqual(L(0, 0), 1.0 - 0.2 - 0.3), "Barycentric 2D L0");
        require(approxEqual(L(1, 0), 0.2), "Barycentric 2D L1");
        require(approxEqual(L(2, 0), 0.3), "Barycentric 2D L2");
        Eigen::MatrixXd back = Cartesian(2, L);
        require(approxEqual(back, xy), "Cartesian(Barycentric(xy)) == xy (2D)");
    }
    // 3D
    {
        Eigen::MatrixXd xyz(3, 2);
        xyz << 0.1, 0.0,
               0.2, 1.0,
               0.3, 0.0;
        Eigen::MatrixXd L = Barycentric(3, xyz);
        require(approxEqual(L(0, 0), 1.0 - 0.1 - 0.2 - 0.3), "Barycentric 3D L0");
        require(approxEqual(L(1, 0), 0.1), "Barycentric 3D L1");
        require(approxEqual(L(2, 0), 0.2), "Barycentric 3D L2");
        require(approxEqual(L(3, 0), 0.3), "Barycentric 3D L3");
        Eigen::MatrixXd back = Cartesian(3, L);
        require(approxEqual(back, xyz), "Cartesian(Barycentric(xyz)) == xyz (3D)");
    }
}

void testCordTransAndMapping2D() {
    // Triangle: (0,0)-(2,0)-(0,3)
    Eigen::MatrixXd V(2, 3);
    V << 0.0, 2.0, 0.0,
         0.0, 0.0, 3.0;

    // Physical vertices should map to reference vertices: (0,0)->(0,0), (2,0)->(1,0), (0,3)->(0,1)
    Eigen::MatrixXd P(2, 3);
    P.col(0) << 0.0, 0.0;
    P.col(1) << 2.0, 0.0;
    P.col(2) << 0.0, 3.0;
    Eigen::MatrixXd rs = CordTrans(V, P);
    require(approxEqual(rs.col(0), (Eigen::Vector2d() << 0.0, 0.0).finished()), "CordTrans V1 -> (0,0)");
    require(approxEqual(rs.col(1), (Eigen::Vector2d() << 1.0, 0.0).finished()), "CordTrans V2 -> (1,0)");
    require(approxEqual(rs.col(2), (Eigen::Vector2d() << 0.0, 1.0).finished()), "CordTrans V3 -> (0,1)");

    // A random point produced from reference (r,s) -> physical should invert back
    Eigen::Vector2d rsv(0.2, 0.3);
    Eigen::MatrixXd bary(3, 1);
    bary << 1.0 - rsv(0) - rsv(1), rsv(0), rsv(1);
    Eigen::MatrixXd phys = V * bary; // physical Cartesian
    Eigen::MatrixXd back = CordTrans(V, phys);
    require(approxEqual(back.col(0), rsv), "CordTrans invertibility 2D");
}

void testCordTransAndMapping3D() {
    // Tetra: (0,0,0)-(2,0,0)-(0,3,0)-(0,0,4)
    Eigen::MatrixXd V(3, 4);
    V << 0.0, 2.0, 0.0, 0.0,
         0.0, 0.0, 3.0, 0.0,
         0.0, 0.0, 0.0, 4.0;

    // Physical vertices -> reference vertices
    Eigen::MatrixXd P(3, 4);
    P.col(0) << 0.0, 0.0, 0.0;
    P.col(1) << 2.0, 0.0, 0.0;
    P.col(2) << 0.0, 3.0, 0.0;
    P.col(3) << 0.0, 0.0, 4.0;
    Eigen::MatrixXd rst = CordTrans(V, P);
    require(approxEqual(rst.col(0), (Eigen::Vector3d() << 0.0, 0.0, 0.0).finished()), "CordTrans V1 -> (0,0,0)");
    require(approxEqual(rst.col(1), (Eigen::Vector3d() << 1.0, 0.0, 0.0).finished()), "CordTrans V2 -> (1,0,0)");
    require(approxEqual(rst.col(2), (Eigen::Vector3d() << 0.0, 1.0, 0.0).finished()), "CordTrans V3 -> (0,1,0)");
    require(approxEqual(rst.col(3), (Eigen::Vector3d() << 0.0, 0.0, 1.0).finished()), "CordTrans V4 -> (0,0,1)");

    // Invertibility check
    Eigen::Vector3d rstv(0.2, 0.3, 0.1);
    Eigen::MatrixXd bary(4, 1);
    bary << 1.0 - rstv(0) - rstv(1) - rstv(2), rstv(0), rstv(1), rstv(2);
    Eigen::MatrixXd phys = V * bary;
    Eigen::MatrixXd back = CordTrans(V, phys);
    require(approxEqual(back.col(0), rstv), "CordTrans invertibility 3D");
}

void testGradientsAndJacobian2D() {
    // Triangle: (0,0)-(a,0)-(0,b)
    const double a = 2.0, b = 3.0;
    Eigen::MatrixXd V(2, 3);
    V << 0.0, a, 0.0,
         0.0, 0.0, b;

    Eigen::MatrixXd G = GradXr(2, V);
    // Should be [[a, 0], [0, b]]
    Eigen::MatrixXd expected(2, 2);
    expected << a, 0.0,
                0.0, b;
    require(approxEqual(G, expected), "GradXr 2D simple right triangle");

    Eigen::MatrixXd R = GradRx(2, V);
    require(approxEqual(R * G, Eigen::MatrixXd::Identity(2, 2)), "GradRx * GradXr == I (2D)");

    double J = Jacobian(2, V);
    require(approxEqual(J, a * b), "Jacobian 2D determinant");
}

void testGradientsAndJacobian3D() {
    // Tetra: (0,0,0)-(a,0,0)-(0,b,0)-(0,0,c)
    const double a = 2.0, b = 3.0, c = 4.0;
    Eigen::MatrixXd V(3, 4);
    V << 0.0, a, 0.0, 0.0,
         0.0, 0.0, b, 0.0,
         0.0, 0.0, 0.0, c;

    Eigen::MatrixXd G = GradXr(3, V);
    Eigen::MatrixXd expected = Eigen::Matrix3d::Zero();
    expected(0, 0) = a;
    expected(1, 1) = b;
    expected(2, 2) = c;
    require(approxEqual(G, expected), "GradXr 3D simple right tetra");

    Eigen::MatrixXd R = GradRx(3, V);
    require(approxEqual(R * G, Eigen::MatrixXd::Identity(3, 3)), "GradRx * GradXr == I (3D)");

    double J = Jacobian(3, V);
    require(approxEqual(J, a * b * c), "Jacobian 3D determinant");
}

void testLagrangianBasisConsistency() {
    // 1D order 2: M * Coef^T = I
    {
        int dim = 1, order = 2;
        Eigen::MatrixXd nodes = EquiNodes(dim, order);
        Eigen::MatrixXd Coef = LagrangianBasis(dim, order, nodes);
        int dof = order + 1;
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dof, dof);
        int idx = 0;
        for (int i = 0; i <= order; ++i) {
            M.col(idx) = nodes.row(0).array().pow(i);
            ++idx;
        }
        Eigen::MatrixXd I = M * Coef.transpose();
        require(approxEqual(I, Eigen::MatrixXd::Identity(dof, dof), 1e-10), "LagrangianBasis 1D order2 interpolation");
    }
    // 2D order 1
    {
        int dim = 2, order = 1;
        Eigen::MatrixXd nodes = EquiNodes(dim, order);
        Eigen::MatrixXd Coef = LagrangianBasis(dim, order, nodes);
        int dof = (order + 1) * (order + 2) / 2;
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dof, dof);
        int idx = 0;
        for (int d = 0; d <= order; ++d) {
            for (int dy = 0; dy <= d; ++dy) {
                int dx = d - dy;
                M.col(idx) = nodes.row(0).array().pow(dx) * nodes.row(1).array().pow(dy);
                ++idx;
            }
        }
        Eigen::MatrixXd I = M * Coef.transpose();
        require(approxEqual(I, Eigen::MatrixXd::Identity(dof, dof), 1e-10), "LagrangianBasis 2D order1 interpolation");
    }
    // 3D order 1
    {
        int dim = 3, order = 1;
        Eigen::MatrixXd nodes = EquiNodes(dim, order);
        Eigen::MatrixXd Coef = LagrangianBasis(dim, order, nodes);
        int dof = (order + 1) * (order + 2) * (order + 3) / 6;
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(dof, dof);
        int idx = 0;
        for (int d = 0; d <= order; ++d) {
            for (int dz = 0; dz <= d; ++dz) {
                for (int dy = 0; dy <= d - dz; ++dy) {
                    int dx = d - dz - dy;
                    M.col(idx) = nodes.row(0).array().pow(dx) * nodes.row(1).array().pow(dy) * nodes.row(2).array().pow(dz);
                    ++idx;
                }
            }
        }
        Eigen::MatrixXd I = M * Coef.transpose();
        require(approxEqual(I, Eigen::MatrixXd::Identity(dof, dof), 1e-10), "LagrangianBasis 3D order1 interpolation");
    }
}

} // namespace

int main() {
    cout << "Running BasisFunctions unit tests..." << endl;

    testEquiNodes();
    testBarycentricCartesian();
    testCordTransAndMapping2D();
    testCordTransAndMapping3D();
    testGradientsAndJacobian2D();
    testGradientsAndJacobian3D();
    testLagrangianBasisConsistency();

    if (failures == 0) {
        cout << "All tests passed." << endl;
    } else {
        cerr << failures << " test(s) failed." << endl;
    }
    return failures;
}


