#include "DGSolver/DGSolver_Advection.hpp"
#include <omp.h>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include "GlobalConfig/GlobalConfig.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "FiniteElement/Integral.hpp"
#include "BoundaryCondition/BCIC.hpp"

using namespace std;
using namespace Eigen;

namespace DGSolver {

    /***************************************************************** 3D ********************************************************************************/

    template<>
    DGSolver_Advection<3>::DGSolver_Advection(alignedVec<MatrixXd>& MassMat, alignedVec<alignedVec<MatrixXd>>& StfMat, 
                                            alignedVec<alignedVec<MatrixXd>>& MassOnFace,
                                            alignedVec<alignedVec<MatrixXd>>& FluxInt,
                                            alignedVec<std::shared_ptr<Element::CellDim<3, 4>>>& Cells,
                                            alignedVec<std::shared_ptr<Element::FaceDim<3, 3>>>& Faces,
                                            std::vector<int>& comp_odr)
    {
        dt = 0.001;
        Vi = Eigen::VectorXd::Zero(3);
        Vi << 1.0, 1.0, 1.0;

        Solution.resize(Cells.size(), VectorXd::Zero(CC.DOF));
        for(int k = 0; k < Cells.size(); k++){
            Solution[k].setZero();
            Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
            for(int i = 0; i < 4; i++){
                Vertices.col(i) = Cells[k]->getVertices()[i]->getCoordinates();
            }
            Eigen::MatrixXd IntPts = phyEquiNodes(CC.SPATIAL_DIM, CC.POLYDEG, Vertices);
            assert(IntPts.rows() == CC.SPATIAL_DIM && IntPts.cols() == CC.DOF);
            Solution[k] = BCIC::EvalF(IntPts, 0, 0, 0);
        }

        cout << "Solution initialized" << endl;

        PreComputedLU.resize(Cells.size(), PartialPivLU<MatrixXd>(MatrixXd::Zero(CC.DOF, CC.DOF)));
        omp_set_num_threads(omp_get_max_threads());
        #pragma omp parallel for
        for(int l = 0; l < CC.N_MESH_CELL; l++){
            int CellID = comp_odr[l];
            cout << "Cell " << CellID << endl;

            MatrixXd Coeff = 1.0 / dt * MassMat[CellID].array();

            for(int i = 0; i < 3; i++){
                Coeff = Coeff.array() - (Vi(i) * StfMat[CellID][i].array());
            }

            for(int il = 0; il < 4; il++){
                int FCID = Cells[CellID]->getFaces()[il]->getIndex();
                int BCTAG = Faces[FCID]->getBoundaryFlag();
                
                if(BCTAG == 0){
                    double f = 0.0;
                    for(int i = 0; i < 3; i++){
                        f += Vi(i) * Cells[CellID]->getOutwardNormVec()[il](i);
                    }
                    Coeff = Coeff.array() + 0.5 * (f + abs(f)) * MassOnFace[CellID][il].array();
                    
                }
                else if(CC.BOUNDARY_COND[BCTAG].first == 1){
                    double f = 0.0;
                    for(int i = 0; i < 3; i++){
                        f += Vi(i) * Cells[CellID]->getOutwardNormVec()[il](i);
                    }
                    Coeff = Coeff.array() + 0.5 * (f + abs(f)) * MassOnFace[CellID][il].array();
                }
            }

            PreComputedLU[CellID] = Coeff.lu();
            cout << "Cell " << CellID << " initialized" << endl;
        }
        cout << "DGSolver_Advection initialized" << endl;
    }

    template<>
    DGSolver_Advection<3>::~DGSolver_Advection() {}

    template<>
    void DGSolver_Advection<3>::solve(alignedVec<std::shared_ptr<Element::FaceDim<3, 3>>>& Faces, 
                                    alignedVec<std::shared_ptr<Element::CellDim<3, 4>>>& Cells, 
                                    std::vector<int>& comp_odr,
                                    alignedVec<MatrixXd>& MassMat, alignedVec<alignedVec<MatrixXd>>& MassOnFace, alignedVec<alignedVec<MatrixXd>>& FluxInt) 
    {
        omp_set_num_threads(omp_get_max_threads());
        // #pragma omp parallel for
        for(int l = 0; l < CC.N_MESH_CELL; l++){
            int CellID = comp_odr[l];

            VectorXd A_src = 1.0 / dt * MassMat[CellID] * Solution[CellID];

            // boundary
            for(int il = 0; il < 4; il++){
                int FCID = Cells[CellID]->getFaces()[il]->getIndex();
                int BCTAG = Faces[FCID]->getBoundaryFlag();

                if(BCTAG == 0){
                    int NBR = (Faces[FCID]->getAdjacentCells()[0] == CellID) ? Faces[FCID]->getAdjacentCells()[1] : Faces[FCID]->getAdjacentCells()[0];
                    double f = 0.0;
                    for(int i = 0; i < 3; i++){
                        f += Vi(i) * Cells[CellID]->getOutwardNormVec()[il](i);
                    }
                    A_src.noalias() -= 0.5 * (f - abs(f)) * (FluxInt[CellID][il] * Solution[NBR]);
                }
                else if(CC.BOUNDARY_COND[BCTAG].first == 1){
                    double f = 0.0;
                    for(int i = 0; i < 3; i++){
                        f += Vi(i) * Cells[CellID]->getOutwardNormVec()[il](i);
                    }
                    A_src.noalias() -= 0.5 * (f - abs(f)) * FluxInt[CellID][il].col(0);
                }
            }

            const auto& lu = PreComputedLU[CellID];
            Solution[CellID] = lu.solve(A_src);
        }

    }

    template<>
    alignedVec<VectorXd> DGSolver_Advection<3>::getSolution() const {
        return Solution;
    }

    template<>
    void DGSolver_Advection<3>::save(alignedVec<std::shared_ptr<Element::CellDim<3, 4>>>& Cells, std::string results_dir, int fix_crd_idx, double crd) {
        std::vector<std::vector<double>> Domain = {{0.0, 1.0}, {0.0, 1.0}};
        std::vector<int> N = {100, 100};

        std::vector<std::vector<int>> CellID(N[0], std::vector<int>(N[1], -1));
        Eigen::MatrixXd P(3, N[0]*N[1]);
        Eigen::MatrixXd refP(3, N[0]*N[1]);
        omp_set_num_threads(omp_get_max_threads());
        if(fix_crd_idx == 0){
            #pragma omp parallel for
            for(int i = 0; i < N[0]; i++){
                for(int j = 0; j < N[1]; j++){
                    P(0, j*N[0] + i) = crd;
                    P(1, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                    P(2, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                }
            }
        }
        else if(fix_crd_idx == 1){
            #pragma omp parallel for
            for(int i = 0; i < N[0]; i++){
                for(int j = 0; j < N[1]; j++){
                    P(0, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                    P(1, j*N[0] + i) = crd;
                    P(2, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                }
            }
        }
        else if(fix_crd_idx == 2){
            #pragma omp parallel for
            for(int i = 0; i < N[0]; i++){
                for(int j = 0; j < N[1]; j++){
                    P(0, j*N[0] + i) = double(i) / double(N[0] - 1) * (Domain[0][1] - Domain[0][0]) + Domain[0][0];
                    P(1, j*N[0] + i) = double(j) / double(N[1] - 1) * (Domain[1][1] - Domain[1][0]) + Domain[1][0];
                    P(2, j*N[0] + i) = crd;
                }
            }
        }

        int cnt = 0;
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                for(int k = 0; k < CC.N_MESH_CELL; k++){
                    if(Cells[k]->isPointInside(std::make_shared<Element::NodeDim<3>>(
                                                            P(0, j*N[0] + i), 
                                                            P(1, j*N[0] + i), 
                                                            P(2, j*N[0] + i))))
                    {
                        CellID[i][j] = k;
                        Eigen::MatrixXd Vertices(3, Cells[k]->getVertices().size());
                        for(int i = 0; i < Cells[k]->getVertices().size(); i++){
                            Vertices.col(i) = Cells[k]->getVertices()[i]->getCoordinates();
                        }
                        refP.col(j*N[0] + i) = CordTrans(Vertices, P.col(j*N[0] + i));
                        cnt++;
                        break;
                    }
                }
            }
        }
        cout << "cnt = " << cnt << endl;


        std::vector<std::vector<double>> ApproxSol(N[0], std::vector<double>(N[1], 0.0));

        std::vector<Polynomial::Polynomial> RefBasis = ReferenceBasis(3, CC.POLYDEG);
        Eigen::MatrixXd Evl(CC.DOF, N[0]*N[1]);
        for(int k = 0; k < RefBasis.size(); k++){
            Evl.row(k) = RefBasis[k].evaluateBatch(refP).transpose();
        }
        #pragma omp parallel for
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                if(CellID[i][j] == -1){
                    continue;
                }
                else{
                    ApproxSol[i][j] = 0.0;
                    for(int l = 0; l < CC.DOF; l++){
                        ApproxSol[i][j] += Solution[CellID[i][j]](l) * Evl(l, j*N[0] + i);
                    }
                }
            }
        }

        ofstream write_output(results_dir);
        assert(write_output.is_open());
        write_output.setf(ios::fixed);
        write_output.precision(16);
        write_output << "x" << " " << "y" << " " << "z" << " " << "ApproxSol" << endl;
        for(int i = 0; i < N[0]; i++){
            for(int j = 0; j < N[1]; j++){
                write_output << P(0, j*N[0] + i) << " " << P(1, j*N[0] + i) << " " << P(2, j*N[0] + i) << " " << ApproxSol[i][j] << endl;
            }
        }
        write_output.close();
    }
}