#pragma once

#include <vector>
#include <fstream>
#include <filesystem>
#include "Eigen/Dense"
#include "SpatialMesh/SpatialMesh.hpp"
#include "Polynomial/polynomial.hpp"
#include "FiniteElement/BasisFunctions.hpp"
#include "FiniteElement/Quadrature.hpp"
#include "Utility/math_utils.hpp"
#include "BoundaryCondition/BCIC.hpp"
#include <iostream>
#include <omp.h>
namespace FiniteElement{
    
    template<int Dim>
    class Integral {
        static_assert(Dim == 2 || Dim == 3, "Mesh dimension must be 2 or 3");
        protected:
            Eigen::MatrixXd IntMat;
            alignedVec<Eigen::MatrixXd> MassMat;
            alignedVec<alignedVec<Eigen::VectorXd>> IntFaceMat;
            alignedVec<alignedVec<Eigen::MatrixXd>> StiffMat;
            alignedVec<alignedVec<Eigen::MatrixXd>> DivDivMat;
            alignedVec<alignedVec<Eigen::MatrixXd>> MassFaceMat;
            alignedVec<alignedVec<alignedVec<Eigen::MatrixXd>>> StiffFaceMat;
            alignedVec<alignedVec<Eigen::MatrixXd>> FluxMat;
            alignedVec<alignedVec<alignedVec<Eigen::MatrixXd>>> StiffFluxMat;
        public:
            explicit Integral(SpatialMesh::SpatialMeshDim<Dim> mesh, int mode); // 0: exact integral; 1: numerical quadrature
            virtual ~Integral(){};

            virtual void updateFlux(SpatialMesh::SpatialMeshDim<Dim> mesh, double dt);
            
            Eigen::MatrixXd getIntMat() const {return IntMat;}
            alignedVec<alignedVec<Eigen::VectorXd>> getIntFaceMat() const {return IntFaceMat;}
            alignedVec<Eigen::MatrixXd> getMassMat() const {return MassMat;}
            alignedVec<alignedVec<Eigen::MatrixXd>> getStiffMat() const {return StiffMat;}
            alignedVec<alignedVec<Eigen::MatrixXd>> getDivDivMat() const {return DivDivMat;}
            alignedVec<alignedVec<Eigen::MatrixXd>> getMassFaceMat() const {return MassFaceMat;}
            alignedVec<alignedVec<alignedVec<Eigen::MatrixXd>>> getStiffFaceMat() const {return StiffFaceMat;}
            alignedVec<alignedVec<Eigen::MatrixXd>> getFluxMat() const {return FluxMat;}
            alignedVec<alignedVec<alignedVec<Eigen::MatrixXd>>> getStiffFluxMat() const {return StiffFluxMat;}
            void save(std::string results_dir);
    };

    template<int Dim>
    Integral<Dim>::Integral(SpatialMesh::SpatialMeshDim<Dim> mesh, int mode){
        int DoF = CC.DOF;
        int Nt = mesh.getNumCells();
        std::vector<double> Volumes(Nt, 0.0);
        for(int i = 0; i < Nt; i++){
            Volumes[i] = mesh.getCells()[i]->getMeasure();
        }
        double alphaVolume, alphaFace;
        if(Dim == 2){
            alphaVolume = 2.0;
            alphaFace = 1.0;
        }
        else if(Dim == 3){
            alphaVolume = 6.0;
            alphaFace = 2.0;
        }
        
        std::vector<Polynomial::Polynomial> RefBasis = ReferenceBasis(Dim, CC.POLYDEG);

        std::cout << ">>> Computing integrals for FEM elementwise operations..." << std::endl;

        /* Compute the integration matrix */
        IntMat.resize(Nt, DoF);
        IntMat.setZero();
        if(mode == 0){
            Eigen::VectorXd IntEle = Utility::int_splx_complete(Dim, CC.POLYDEG);
            double s = 0.0;
            for (int i = 0; i < DoF; i++){
                s = RefBasis[i].getCoefficients().dot(IntEle);
                for (int k = 0; k < Nt; k++){ 
                    IntMat(k, i) = s * alphaVolume * Volumes[k];
                }
            }
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad3D Quad3D(14);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                Eigen::MatrixXd QuadCords = Quad3D.Cords(Vertices);
                for(int i = 0; i < DoF; i++){
                    IntMat(k, i) = Quad3D.Int(LB[i].evaluateBatch(QuadCords)) * alphaVolume * Volumes[k];
                }
            }
        }
        std::cout << "  >>> IntMat computed." << std::endl;

        /* Compute the mass matrix */
        MassMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            MassMat[i].resize(DoF, DoF);
            MassMat[i].setZero();
        }
        if(mode == 0){
            Eigen::VectorXd IntEle = Utility::int_splx_complete(Dim, CC.POLYDEG + CC.POLYDEG);
            for(int j = 0; j < DoF; j++){
                for(int k = 0; k < DoF; k++){
                    Polynomial::Polynomial product = RefBasis[j] * RefBasis[k];
                    for(int i = 0; i < Nt; i++){
                        MassMat[i](j, k) = product.getCoefficients().dot(IntEle) * alphaVolume * Volumes[i];
                    }
                }
            }
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad3D Quad3D(46);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                Eigen::MatrixXd QuadCords = Quad3D.Cords(Vertices);
                #pragma omp parallel for
                for(int i = 0; i < DoF; i++){
                    for(int j = 0; j < DoF; j++){
                        Polynomial::Polynomial product = LB[i] * LB[j];
                        MassMat[k](i, j) = Quad3D.Int(product.evaluateBatch(QuadCords)) * alphaVolume * Volumes[k];
                    }
                }
            }
        }

        std::cout << "  >>> MassMat computed." << std::endl;

        /* Compute the stiffness matrix */
        StiffMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            StiffMat[i].resize(Dim);
            for(int j = 0; j < Dim; j++){
                StiffMat[i][j].resize(DoF, DoF);
                StiffMat[i][j].setZero();
            }
        }
        if(mode == 0){
            alignedVec<Eigen::MatrixXd> refStifMat(Dim, Eigen::MatrixXd::Zero(DoF, DoF));
            Eigen::VectorXd IntEleDer = Utility::int_splx_complete(Dim, CC.POLYDEG + CC.POLYDEG - 1);
            double s = 0.0;
            for(int j = 0; j < DoF; j++){
                for(int k = 0; k < DoF; k++){
                    for(int l = 0; l < Dim; l++){
                        refStifMat[l](j, k) = (RefBasis[j].derivative(l) * RefBasis[k]).getCoefficients().dot(IntEleDer);              
                    }
                    for(int i = 0; i < Nt; i++){
                        Eigen::MatrixXd vertices = Eigen::MatrixXd::Zero(Dim, Dim+1);
                        for(int l = 0; l < Dim+1; l++){
                            vertices.col(l) = mesh.getCells()[i]->getVertices()[l]->getCoordinates();
                        }
                        s = Jacobian(Dim, vertices);
                        Eigen::MatrixXd Xr = GradXr(Dim, vertices);
                        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(Dim, Dim);
                        if(Dim == 2){
                            J(0, 0) = Xr(1,1) / s;
                            J(0, 1) = -Xr(1,0) / s;
                            J(1, 0) = -Xr(0,1) / s;
                            J(1, 1) = Xr(0,0) / s;
                        }
                        else if(Dim == 3){
                            J(0, 0) = (Xr(1,1) * Xr(2,2) - Xr(2,1) * Xr(1,2)) / s;
                            J(0, 1) = - (Xr(1,0) * Xr(2,2) - Xr(2, 0) * Xr(1,2)) / s;
                            J(0, 2) = (Xr(1,0) * Xr(2,1) - Xr(2,0) * Xr(1,1)) / s;
                            J(1, 0) = - (Xr(0,1) * Xr(2,2) - Xr(2,1) * Xr(0,2)) / s;
                            J(1, 1) = (Xr(0,0) * Xr(2,2) - Xr(2,0) * Xr(0,2)) / s;
                            J(1, 2) = - (Xr(0,0) * Xr(2,1) - Xr(2,0) * Xr(0,1)) / s;
                            J(2, 0) = (Xr(0,1) * Xr(1,2) - Xr(1,1) * Xr(0,2)) / s;
                            J(2, 1) = - (Xr(0,0) * Xr(1,2) - Xr(1,0) * Xr(0,2)) / s;
                            J(2, 2) = (Xr(0,0) * Xr(1,1) - Xr(1,0) * Xr(0,1)) / s;
                        }
                        for(int l = 0; l < Dim; l++){
                            for(int m = 0; m < Dim; m++){
                                StiffMat[i][l](j, k) += J(l, m) * refStifMat[m](j, k) * alphaVolume * Volumes[i];
                            }
                        }
                    }
                }
            }
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad3D Quad3D(46);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                Eigen::MatrixXd QuadCords = Quad3D.Cords(Vertices);
                #pragma omp parallel for
                for(int i = 0; i < DoF; i++){
                    for(int j = 0; j < DoF; j++){
                        for(int l = 0; l < Dim; l++){
                            Polynomial::Polynomial product = LB[i].derivative(l) * LB[j];
                            StiffMat[k][l](i, j) = Quad3D.Int(product.evaluateBatch(QuadCords)) * alphaVolume * Volumes[k];
                        }
                    }
                }
            }
        }

        std::cout << "  >>> StiffMat computed." << std::endl;

        /* Compute the div-div matrix */
        DivDivMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            DivDivMat[i].resize(Dim);
            for(int j = 0; j < Dim; j++){
                DivDivMat[i][j].resize(DoF, DoF);
                DivDivMat[i][j].setZero();
            }
        }
        if(mode == 0){
            throw std::runtime_error("Exact integral is not supported for div-div matrix");
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad3D Quad3D(46);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                Eigen::MatrixXd QuadCords = Quad3D.Cords(Vertices);
                #pragma omp parallel for
                for(int i = 0; i < DoF; i++){
                    for(int j = 0; j < DoF; j++){
                        for(int l = 0; l < Dim; l++){
                            Polynomial::Polynomial product = LB[i].derivative(l) * LB[j].derivative(l);
                            DivDivMat[k][l](i, j) = Quad3D.Int(product.evaluateBatch(QuadCords)) * alphaVolume * Volumes[k];
                        }
                    }
                }
            }
        }

        /* Compute the integration matrix on the faces */
        IntFaceMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            IntFaceMat[i].resize(Dim+1);
            for(int j = 0; j < Dim+1; j++){
                IntFaceMat[i][j].resize(DoF);
                IntFaceMat[i][j].setZero();
            }
        }
        if(mode == 0){
            alignedVec<Eigen::VectorXd> IntEleFace = alignedVec<Eigen::VectorXd>(Dim+1, Eigen::VectorXd::Zero(DoF));
            int idx = 0;
            if(Dim == 2){
                for(int d = 0; d <= CC.POLYDEG; d++){
                    for(int dy = 0; dy <= d; dy++){
                        int dx = d - dy;
                        if(dx == 0){
                            IntEleFace[2](idx) = Utility::int_splx_mono(1, {dy});
                        }
                        if(dy == 0){
                            IntEleFace[0](idx) = Utility::int_splx_mono(1, {dx});
                        }
                        IntEleFace[1](idx) = Utility::int_splx_mono(1, {dx, dy});
                        idx ++;
                    }
                }
            }
            else if(Dim == 3){
                for(int d = 0; d <= CC.POLYDEG; d++){
                    for(int dx = d; dx >= 0; dx--){
                        for(int dy = d-dx; dy >= 0; dy--){
                            int dz = d - dx - dy;
                            if(dx == 0){
                                IntEleFace[2](idx) = Utility::int_splx_mono(2, {dy, dz});
                            }
                            if(dy == 0){
                                IntEleFace[3](idx) = Utility::int_splx_mono(2, {dx, dz});
                            }
                            if(dz == 0){
                                IntEleFace[0](idx) = Utility::int_splx_mono(2, {dx, dy});
                            }
                            IntEleFace[1](idx) = Utility::int_splx_mono(2, {dx, dy, dz});
                            idx ++;
                        }
                    }
                }
            }
            for(int j = 0; j < DoF; j++){
                for(int i = 0; i < Nt; i++){
                    for(int l = 0; l < Dim + 1; l++){
                        IntFaceMat[i][l](j) = RefBasis[j].getCoefficients().dot(IntEleFace[l]) * alphaFace * mesh.getCells()[i]->getFaces()[l]->getMeasure();
                    }
                }
            }
        }
        else if (mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad2D Quad2D(16);
            #pragma omp parallel for
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                for(int l = 0; l < 4; l++){
                    Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                    for(int m = 0; m < 3; m++){
                        FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                    }
                    Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                    for(int i = 0; i < DoF; i++){
                        IntFaceMat[k][l](i) = Quad2D.Int(LB[i].evaluateBatch(QuadCords)) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                    }
                }
            }
        }

        std::cout << "  >>> IntFaceMat computed." << std::endl;

        /* Compute the mass matrix on the faces */
        MassFaceMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            MassFaceMat[i].resize(Dim+1);
            for(int j = 0; j < Dim+1; j++){
                MassFaceMat[i][j].resize(DoF, DoF);
                MassFaceMat[i][j].setZero();
            }
        }
        if(mode == 0){
            int dof2;
            if(Dim == 2){
                dof2 = (CC.POLYDEG + CC.POLYDEG + 1) * (CC.POLYDEG + CC.POLYDEG + 2) / 2;
            }
            else if(Dim == 3){
                dof2 = (CC.POLYDEG + CC.POLYDEG + 1) * (CC.POLYDEG + CC.POLYDEG + 2) * (CC.POLYDEG + CC.POLYDEG + 3) / 6;
            }
            alignedVec<Eigen::VectorXd> IntEleMassFace = alignedVec<Eigen::VectorXd>(Dim+1, Eigen::VectorXd::Zero(dof2));
            int idx = 0;
            if(Dim == 2){
                for(int d = 0; d <= CC.POLYDEG + CC.POLYDEG; d++){
                    for(int dy = 0; dy <= d; dy++){
                        int dx = d - dy;
                        if(dx == 0){
                            IntEleMassFace[2](idx) = Utility::int_splx_mono(1, {dy});
                        }
                        if(dy == 0){
                            IntEleMassFace[0](idx) = Utility::int_splx_mono(1, {dx});
                        }
                        IntEleMassFace[1](idx) = Utility::int_splx_mono(1, {dx, dy});
                        idx ++;
                    }
                }
            }
            else if(Dim == 3){
                for(int d = 0; d <= CC.POLYDEG + CC.POLYDEG; d++){
                    for(int dx = d; dx >= 0; dx--){
                        for(int dy = d-dx; dy >= 0; dy--){
                            int dz = d - dx - dy;
                            if(dx == 0){
                                IntEleMassFace[2](idx) = Utility::int_splx_mono(2, {dy, dz});
                            }
                            if(dy == 0){
                                IntEleMassFace[3](idx) = Utility::int_splx_mono(2, {dx, dz});
                            }
                            if(dz == 0){
                                IntEleMassFace[0](idx) = Utility::int_splx_mono(2, {dx, dy});
                            }
                            IntEleMassFace[1](idx) = Utility::int_splx_mono(2, {dx, dy, dz});
                            idx ++;
                        }
                    }
                }
            }
            for(int j = 0; j < DoF; j++){
                for(int k = 0; k < DoF; k++){
                    Polynomial::Polynomial product = RefBasis[j] * RefBasis[k];
                    for(int i = 0; i < Nt; i++){
                        for(int l = 0; l < Dim + 1; l++){
                            MassFaceMat[i][l](j, k) = product.getCoefficients().dot(IntEleMassFace[l]) * alphaFace * mesh.getCells()[i]->getFaces()[l]->getMeasure();
                        }
                    }
                }
            }
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad2D Quad2D(16);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                for(int l = 0; l < 4; l++){
                    Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                    for(int m = 0; m < 3; m++){
                        FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                    }
                    Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                    #pragma omp parallel for
                    for(int i = 0; i < DoF; i++){
                        for(int j = 0; j < DoF; j++){
                            Polynomial::Polynomial product = LB[i] * LB[j];
                            MassFaceMat[k][l](i, j) = Quad2D.Int(product.evaluateBatch(QuadCords)) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                        }
                    }
                }
            }
        }

        std::cout << "  >>> MassFaceMat computed." << std::endl;

        /* Compute the stiffness matrix on the faces */
        StiffFaceMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            StiffFaceMat[i].resize(Dim+1);
            for(int j = 0; j < Dim+1; j++){
                StiffFaceMat[i][j].resize(Dim);
                for(int k = 0; k < Dim; k++){
                    StiffFaceMat[i][j][k].resize(DoF, DoF);
                    StiffFaceMat[i][j][k].setZero();
                }
            }
        }
        if(mode == 0){
            throw std::runtime_error("Exact integral is not supported for stiffness matrix on the faces");
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad2D Quad2D(16);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    Vertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> LB = LagrangeBasis(3, CC.POLYDEG, Vertices);
                for(int l = 0; l < 4; l++){
                    Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                    for(int m = 0; m < 3; m++){
                        FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                    }
                    Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                    #pragma omp parallel for
                    for(int i = 0; i < DoF; i++){
                        for(int j = 0; j < DoF; j++){
                            for(int d = 0; d < Dim; d++){
                                Polynomial::Polynomial product = LB[i].derivative(d) * LB[j];
                                StiffFaceMat[k][l][d](i, j) = Quad2D.Int(product.evaluateBatch(QuadCords)) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                            }
                        }
                    }
                }
            }
        }

        std::cout << "  >>> StiffFaceMat computed." << std::endl;

        /* Compute the product of basis functions from both sides of the faces */
        FluxMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            FluxMat[i].resize(Dim+1);
            for(int j = 0; j < Dim+1; j++){
                FluxMat[i][j].resize(DoF, DoF);
                FluxMat[i][j].setZero();
            }
        }
        if(mode == 0){
            Eigen::MatrixXd Trans = Eigen::MatrixXd::Zero(DoF, DoF);
            Eigen::MatrixXd Alpha = Eigen::MatrixXd::Zero(DoF, DoF);
            for(int i = 0; i < Nt; i++){
                auto curCell = std::dynamic_pointer_cast<Element::CellDim<Dim, Dim+1>>(mesh.getCells()[i]);
                Eigen::MatrixXd curVertices = Eigen::MatrixXd::Zero(Dim, Dim+1);
                for(int l = 0; l < Dim+1; l++){
                    curVertices.col(l) = curCell->getVertices()[l]->getCoordinates();
                }
                Eigen::MatrixXd curPhysicalNodes = phyEquiNodes(Dim, CC.POLYDEG, curVertices);
                for(int j = 0; j < Dim+1; j++){
                    int adjCellIndex = curCell->getAdjacentCell(j);
                    if (adjCellIndex != -1) {
                        auto adjCell = std::dynamic_pointer_cast<Element::CellDim<Dim, Dim+1>>(mesh.getCells()[adjCellIndex]);
                        if (adjCell) {
                            Eigen::MatrixXd adjVertices = Eigen::MatrixXd::Zero(Dim, Dim+1);
                            for(int l = 0; l < Dim+1; l++){
                                adjVertices.col(l) = adjCell->getVertices()[l]->getCoordinates();
                            }
                            Trans = CordTrans(adjVertices, curPhysicalNodes);
                            for(int k = 0; k < RefBasis.size(); k++){
                                Alpha.row(k) = RefBasis[k].evaluateBatch(Trans).transpose();
                            }
                            FluxMat[i][j] = (Alpha * MassFaceMat[i][j]).transpose();
                        }
                    }
                    else if(CC.BOUNDARY_COND[curCell->getFaces()[j]->getBoundaryFlag()].first == 1){
                        for(int k = 0; k < DoF; k++){
                            FluxMat[i][j].col(k) = IntFaceMat[i][j];
                        }
                    }
                }
            }
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad2D Quad2D(16);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd curVertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    curVertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> curLB = LagrangeBasis(3, CC.POLYDEG, curVertices);
                for(int l = 0; l < 4; l++){
                    int FCID = mesh.getCells()[k]->getFaces()[l]->getIndex();
                    int BCTAG = mesh.getCells()[k]->getFaces()[l]->getBoundaryFlag();
                    if(BCTAG == 0){
                        int NBR = (mesh.getFaces()[FCID]->getAdjacentCells()[0] == k) ? mesh.getFaces()[FCID]->getAdjacentCells()[1] : mesh.getFaces()[FCID]->getAdjacentCells()[0];
                        Eigen::MatrixXd adjVertices = Eigen::MatrixXd::Zero(3, 4);
                        for(int m = 0; m < 4; m++){
                            adjVertices.col(m) = mesh.getCells()[NBR]->getVertices()[m]->getCoordinates();
                        }
                        std::vector<Polynomial::Polynomial> adjLB = LagrangeBasis(3, CC.POLYDEG, adjVertices);
                        Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                        for(int m = 0; m < 3; m++){
                            FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                        }
                        Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                        #pragma omp parallel for
                        for(int i = 0; i < DoF; i++){
                            for(int j = 0; j < DoF; j++){
                                Polynomial::Polynomial product = curLB[i] * adjLB[j];
                                FluxMat[k][l](i, j) = Quad2D.Int(product.evaluateBatch(QuadCords)) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                            }
                        }
                    }
                    else if(CC.BOUNDARY_COND[BCTAG].first == 1){
                        Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                        for(int m = 0; m < 3; m++){
                            FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                        }
                        Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                        #pragma omp parallel for
                        for(int i = 0; i < DoF; i++){
                            for(int j = 0; j < DoF; j++){
                                Eigen::VectorXd f = curLB[i].evaluateBatch(QuadCords);
                                f = f.array() * BCIC::EvalF(QuadCords, 0, BCTAG, 0).array();
                                FluxMat[k][l](i, j) = Quad2D.Int(f) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                            }
                        }
                    }
                }
            }
        }

        std::cout << "  >>> FluxMat computed." << std::endl;

        StiffFluxMat.resize(Nt);
        for(int i = 0; i < Nt; i++){
            StiffFluxMat[i].resize(Dim+1);
            for(int j = 0; j < Dim+1; j++){
                StiffFluxMat[i][j].resize(Dim);
                for(int k = 0; k < Dim; k++){
                    StiffFluxMat[i][j][k].resize(DoF, DoF);
                    StiffFluxMat[i][j][k].setZero();
                }
            }
        }
        if(mode == 0){
            throw std::runtime_error("Exact integral is not supported for stiffness matrix on the faces");
        }
        else if(mode == 1){
            if(Dim == 2){
                throw std::runtime_error("Numerical quadrature is not supported for 2D mesh");
            }
            FiniteElement::Quad2D Quad2D(16);
            for(int k = 0; k < Nt; k++){
                Eigen::MatrixXd curVertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 0; i < 4; i++){
                    curVertices.col(i) = mesh.getCells()[k]->getVertices()[i]->getCoordinates();
                }
                std::vector<Polynomial::Polynomial> curLB = LagrangeBasis(3, CC.POLYDEG, curVertices);
                for(int l = 0; l < 4; l++){
                    int FCID = mesh.getCells()[k]->getFaces()[l]->getIndex();
                    int BCTAG = mesh.getCells()[k]->getFaces()[l]->getBoundaryFlag();
                    if(BCTAG == 0){
                        int NBR = (mesh.getFaces()[FCID]->getAdjacentCells()[0] == k) ? mesh.getFaces()[FCID]->getAdjacentCells()[1] : mesh.getFaces()[FCID]->getAdjacentCells()[0];
                        Eigen::MatrixXd adjVertices = Eigen::MatrixXd::Zero(3, 4);
                        for(int m = 0; m < 4; m++){
                            adjVertices.col(m) = mesh.getCells()[NBR]->getVertices()[m]->getCoordinates();
                        }
                        std::vector<Polynomial::Polynomial> adjLB = LagrangeBasis(3, CC.POLYDEG, adjVertices);
                        Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                        for(int m = 0; m < 3; m++){
                            FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                        }
                        Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                        #pragma omp parallel for
                        for(int i = 0; i < DoF; i++){
                            for(int j = 0; j < DoF; j++){
                                for(int d = 0; d < Dim; d++){
                                    Polynomial::Polynomial product = curLB[i].derivative(d) * adjLB[j];
                                    StiffFluxMat[k][l][d](i, j) = Quad2D.Int(product.evaluateBatch(QuadCords)) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                                }
                            }
                        }
                    }
                    else if(CC.BOUNDARY_COND[BCTAG].first == 1){
                        Eigen::MatrixXd FaceVertices = Eigen::MatrixXd::Zero(3, 3);
                        for(int m = 0; m < 3; m++){
                            FaceVertices.col(m) = mesh.getCells()[k]->getFaces()[l]->getVertices()[m]->getCoordinates();
                        }
                        Eigen::MatrixXd QuadCords = Quad2D.Cords(FaceVertices);
                        #pragma omp parallel for
                        for(int i = 0; i < DoF; i++){
                            for(int d = 0; d < Dim; d++){
                                Eigen::VectorXd f = curLB[i].derivative(d).evaluateBatch(QuadCords);
                                f = f.array() * BCIC::EvalF(QuadCords, 0, BCTAG, 0).array();
                                for(int j = 0; j < DoF; j++){
                                    StiffFluxMat[k][l][d](i, j) = Quad2D.Int(f) * alphaFace * mesh.getCells()[k]->getFaces()[l]->getMeasure();
                                }
                            }
                        }
                    }
                }
            }
        }

        std::cout << "  >>> StiffFluxMat computed." << std::endl;

        std::cout << ">>> FEM elementwise operations completed." << std::endl;
    }

    template<int Dim>
    void Integral<Dim>::updateFlux(SpatialMesh::SpatialMeshDim<Dim> mesh, double dt){
        for(int i = 0; i < CC.N_MESH_CELL; i++){
            for(int j = 0; j < Dim+1; j++){
                int BCTAG = mesh.getCells()[i]->getFaces()[j]->getBoundaryFlag();
                if(BCTAG == 0){
                    continue;
                }
                else if(CC.BOUNDARY_COND[BCTAG].first == 1){
                    for(int d = 0; d < Dim; d++){
                        StiffFluxMat[i][j][d] = StiffFluxMat[i][j][d].array() * exp(dt);
                    }
                    FluxMat[i][j] = FluxMat[i][j].array() * exp(dt);
                }
            }
        }
    }

    template<int Dim>
    void Integral<Dim>::save(std::string results_dir){
        std::cout << ">>> Saving Integral matrices to files..." << std::endl;
        
        // Create results directory if it doesn't exist
        // std::string results_dir = "output/logs/log3D";
        std::filesystem::create_directories(results_dir);
        
        // Save IntMat
        std::ofstream intmat_file(results_dir + "/IntMat.txt");
        if (intmat_file.is_open()) {
            intmat_file << "# Integration Matrix (IntMat)" << std::endl;
            intmat_file << "# Dimensions: " << IntMat.rows() << " x " << IntMat.cols() << std::endl;
            intmat_file << "# Format: Matrix in Eigen format" << std::endl;
            intmat_file << IntMat << std::endl;
            intmat_file.close();
            std::cout << "  >>> IntMat saved to " << results_dir << "/IntMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for IntMat" << std::endl;
        }

        // Save MassMat
        std::ofstream massmat_file(results_dir + "/MassMat.txt");
        if (massmat_file.is_open()) {
            massmat_file << "# Mass Matrix (MassMat)" << std::endl;
            massmat_file << "# Number of elements: " << MassMat.size() << std::endl;
            if (!MassMat.empty()) {
                massmat_file << "# Element matrix dimensions: " << MassMat[0].rows() << " x " << MassMat[0].cols() << std::endl;
            }
            massmat_file << "# Format: Each element matrix separated by '---'" << std::endl;
            for (size_t i = 0; i < MassMat.size(); ++i) {
                massmat_file << "# Element " << i << std::endl;
                massmat_file << MassMat[i] << std::endl;
                if (i < MassMat.size() - 1) {
                    massmat_file << "---" << std::endl;
                }
            }
            massmat_file.close();
            std::cout << "  >>> MassMat saved to " << results_dir << "/MassMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for MassMat" << std::endl;
        }


        // Save IntFaceMat
        std::ofstream intfacemat_file(results_dir + "/IntFaceMat.txt");
        if (intfacemat_file.is_open()) {
            intfacemat_file << "# Integration Face Matrix (IntFaceMat)" << std::endl;
            intfacemat_file << "# Number of elements: " << IntFaceMat.size() << std::endl;
            if (!IntFaceMat.empty() && !IntFaceMat[0].empty()) {
                intfacemat_file << "# Number of faces per element: " << IntFaceMat[0].size() << std::endl;
                intfacemat_file << "# Face vector length: " << IntFaceMat[0][0].size() << std::endl;
            }
            intfacemat_file << "# Format: Element -> Face -> Vector, separated by '---'" << std::endl;
            for (size_t i = 0; i < IntFaceMat.size(); ++i) {
                intfacemat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < IntFaceMat[i].size(); ++j) {
                    intfacemat_file << "# Face " << j << std::endl;
                    intfacemat_file << IntFaceMat[i][j].transpose() << std::endl;
                    if (j < IntFaceMat[i].size() - 1) {
                        intfacemat_file << "---" << std::endl;
                    }
                }
                if (i < IntFaceMat.size() - 1) {
                    intfacemat_file << "===" << std::endl;
                }
            }
            intfacemat_file.close();
            std::cout << "  >>> IntFaceMat saved to " << results_dir << "/IntFaceMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for IntFaceMat" << std::endl;
        }

        // Save StiffMat
        std::ofstream stiffmat_file(results_dir + "/StiffMat.txt");
        if (stiffmat_file.is_open()) {
            stiffmat_file << "# Stiffness Matrix (StiffMat)" << std::endl;
            stiffmat_file << "# Number of elements: " << StiffMat.size() << std::endl;
            if (!StiffMat.empty() && !StiffMat[0].empty()) {
                stiffmat_file << "# Number of dimensions: " << StiffMat[0].size() << std::endl;
                stiffmat_file << "# Element matrix dimensions: " << StiffMat[0][0].rows() << " x " << StiffMat[0][0].cols() << std::endl;
            }
            stiffmat_file << "# Format: Element -> Dimension -> Matrix, separated by '---'" << std::endl;
            for (size_t i = 0; i < StiffMat.size(); ++i) {
                stiffmat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < StiffMat[i].size(); ++j) {
                    stiffmat_file << "# Dimension " << j << std::endl;
                    stiffmat_file << StiffMat[i][j] << std::endl;
                    if (j < StiffMat[i].size() - 1) {
                        stiffmat_file << "---" << std::endl;
                    }
                }
                if (i < StiffMat.size() - 1) {
                    stiffmat_file << "===" << std::endl;
                }
            }
            stiffmat_file.close();
            std::cout << "  >>> StiffMat saved to " << results_dir << "/StiffMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for StiffMat" << std::endl;
        }

        
        // Save MassFaceMat
        std::ofstream massfacemat_file(results_dir + "/MassFaceMat.txt");
        if (massfacemat_file.is_open()) {
            massfacemat_file << "# Mass Face Matrix (MassFaceMat)" << std::endl;
            massfacemat_file << "# Number of elements: " << MassFaceMat.size() << std::endl;
            if (!MassFaceMat.empty() && !MassFaceMat[0].empty()) {
                massfacemat_file << "# Number of faces per element: " << MassFaceMat[0].size() << std::endl;
                massfacemat_file << "# Face matrix dimensions: " << MassFaceMat[0][0].rows() << " x " << MassFaceMat[0][0].cols() << std::endl;
            }
            massfacemat_file << "# Format: Element -> Face -> Matrix, separated by '---'" << std::endl;
            for (size_t i = 0; i < MassFaceMat.size(); ++i) {
                massfacemat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < MassFaceMat[i].size(); ++j) {
                    massfacemat_file << "# Face " << j << std::endl;
                    massfacemat_file << MassFaceMat[i][j] << std::endl;
                    if (j < MassFaceMat[i].size() - 1) {
                        massfacemat_file << "---" << std::endl;
                    }
                }
                if (i < MassFaceMat.size() - 1) {
                    massfacemat_file << "===" << std::endl;
                }
            }
            massfacemat_file.close();
            std::cout << "  >>> MassFaceMat saved to " << results_dir << "/MassFaceMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for MassFaceMat" << std::endl;
        }

        
        // Save FluxMat
        std::ofstream fluxmat_file(results_dir + "/FluxMat.txt");
        if (fluxmat_file.is_open()) {
            fluxmat_file << "# Flux Matrix (FluxMat)" << std::endl;
            fluxmat_file << "# Number of elements: " << FluxMat.size() << std::endl;
            if (!FluxMat.empty() && !FluxMat[0].empty()) {
                fluxmat_file << "# Number of faces per element: " << FluxMat[0].size() << std::endl;
                fluxmat_file << "# Face matrix dimensions: " << FluxMat[0][0].rows() << " x " << FluxMat[0][0].cols() << std::endl;
            }
            fluxmat_file << "# Format: Element -> Face -> Matrix, separated by '---'" << std::endl;
            for (size_t i = 0; i < FluxMat.size(); ++i) {
                fluxmat_file << "# Element " << i << std::endl;
                for (size_t j = 0; j < FluxMat[i].size(); ++j) {
                    fluxmat_file << "# Face " << j << std::endl;
                    fluxmat_file << FluxMat[i][j] << std::endl;
                    if (j < FluxMat[i].size() - 1) {
                        fluxmat_file << "---" << std::endl;
                    }
                }
                if (i < FluxMat.size() - 1) {
                    fluxmat_file << "===" << std::endl;
                }
            }
            fluxmat_file.close();
            std::cout << "  >>> FluxMat saved to " << results_dir << "/FluxMat.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for FluxMat" << std::endl;
        }
        
        
        // Save metadata summary
        std::ofstream metadata_file(results_dir + "/Integral_metadata.txt");
        if (metadata_file.is_open()) {
            metadata_file << "# Integral Matrix Metadata" << std::endl;
            metadata_file << "# Generated on: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
            metadata_file << "# Mode: " << CC.QUAD_MODE << std::endl;
            metadata_file << std::endl;
            
            metadata_file << "IntMat:" << std::endl;
            metadata_file << "  Dimensions: " << IntMat.rows() << " x " << IntMat.cols() << std::endl;
            metadata_file << "  Size: " << IntMat.size() << " elements" << std::endl;
            metadata_file << std::endl;
            
            metadata_file << "MassMat:" << std::endl;
            metadata_file << "  Number of elements: " << MassMat.size() << std::endl;
            if (!MassMat.empty()) {
                metadata_file << "  Matrix dimensions: " << MassMat[0].rows() << " x " << MassMat[0].cols() << std::endl;
                metadata_file << "  Total size: " << MassMat.size() * MassMat[0].size() << " elements" << std::endl;
            }
            metadata_file << std::endl;
            
            metadata_file << "StiffMat:" << std::endl;
            metadata_file << "  Number of elements: " << StiffMat.size() << std::endl;
            if (!StiffMat.empty() && !StiffMat[0].empty()) {
                metadata_file << "  Dimensions per element: " << StiffMat[0].size() << std::endl;
                metadata_file << "  Matrix dimensions: " << StiffMat[0][0].rows() << " x " << StiffMat[0][0].cols() << std::endl;
                metadata_file << "  Total size: " << StiffMat.size() * StiffMat[0].size() * StiffMat[0][0].size() << " elements" << std::endl;
            }
            metadata_file << std::endl;
            
            metadata_file << "MassFaceMat:" << std::endl;
            metadata_file << "  Number of elements: " << MassFaceMat.size() << std::endl;
            if (!MassFaceMat.empty() && !MassFaceMat[0].empty()) {
                metadata_file << "  Faces per element: " << MassFaceMat[0].size() << std::endl;
                metadata_file << "  Matrix dimensions: " << MassFaceMat[0][0].rows() << " x " << MassFaceMat[0][0].cols() << std::endl;
                metadata_file << "  Total size: " << MassFaceMat.size() * MassFaceMat[0].size() * MassFaceMat[0][0].size() << " elements" << std::endl;
            }
            metadata_file << std::endl;
            
            metadata_file << "FluxMat:" << std::endl;
            metadata_file << "  Number of elements: " << FluxMat.size() << std::endl;
            if (!FluxMat.empty() && !FluxMat[0].empty()) {
                metadata_file << "  Faces per element: " << FluxMat[0].size() << std::endl;
                metadata_file << "  Matrix dimensions: " << FluxMat[0][0].rows() << " x " << FluxMat[0][0].cols() << std::endl;
                metadata_file << "  Total size: " << FluxMat.size() * FluxMat[0].size() * FluxMat[0][0].size() << " elements" << std::endl;
            }
            
            metadata_file.close();
            std::cout << "  >>> Metadata saved to " << results_dir << "/Integral_metadata.txt" << std::endl;
        } else {
            std::cerr << "  >>> Error: Could not open file for metadata" << std::endl;
        }
        
        std::cout << ">>> Integral matrices saved successfully to " << results_dir << "/ directory" << std::endl;
    }

}