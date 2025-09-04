#pragma once

#include <Eigen/Dense>
#include <omp.h>
#include "GlobalConfig/GlobalConfig.hpp"
#include "DGSolver/DGSolver.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "FiniteElement/Integral.hpp"
#include "Utility/math_utils.hpp"

using namespace Eigen;
using namespace std;

namespace DGSolver {

    template<int Dim>
    class DGSolver_Advection : public DGSolver<Dim> {
        private:
            Eigen::VectorXd Vi;
            double dt;
            alignedVec<VectorXd> Solution;
            alignedVec<PartialPivLU<MatrixXd>> PreComputedLU;
        public:
            DGSolver_Advection(alignedVec<MatrixXd>& MassMat, alignedVec<alignedVec<MatrixXd>>& StfMat, 
                              alignedVec<alignedVec<MatrixXd>>& MassOnFace,
                              alignedVec<alignedVec<MatrixXd>>& FluxInt,
                              alignedVec<std::shared_ptr<Element::CellDim<Dim, Dim+1>>>& Cells,
                              alignedVec<std::shared_ptr<Element::FaceDim<Dim, Dim>>>& Faces,
                              std::vector<int>& comp_odr);
            ~DGSolver_Advection();
            void solve(alignedVec<std::shared_ptr<Element::FaceDim<Dim, Dim>>>& Faces, 
                       alignedVec<std::shared_ptr<Element::CellDim<Dim, Dim+1>>>& Cells, 
                       std::vector<int>& comp_odr,
                       alignedVec<MatrixXd>& MassMat, alignedVec<alignedVec<MatrixXd>>& MassOnFace, alignedVec<alignedVec<MatrixXd>>& FluxInt);
            alignedVec<VectorXd> getSolution() const;
            double get_dt() const {return dt;}
            void save(alignedVec<std::shared_ptr<Element::CellDim<3, 4>>>& Cells, std::string results_dir, int fix_crd_idx, double crd);
    };

} // namespace DGSolver