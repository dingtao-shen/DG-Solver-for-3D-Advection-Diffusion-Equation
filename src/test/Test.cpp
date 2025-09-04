#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <omp.h>
#include <iomanip>
#include <memory>
#include <stdexcept>

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "GlobalConfig/GlobalConfig.hpp"
#include "Utility/math_utils.hpp"
#include "Polynomial/polynomial.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "FiniteElement/Integral.hpp"

#include "DGSolver/DGSolver_Advection.hpp"

using namespace Eigen;
using namespace std;
using namespace Polynomial;

ControlConstants CC;

#ifndef DIM
#define DIM 2
#endif

int main(int argc, char* argv[]){
    cout.precision(12);

    CC.LoadFromFile("config/control/Control.yaml");
    cout << "  >>> Parameter loading completed successfully!" << endl;


    SpatialMesh::SpatialMeshDim<DIM> mesh(CC.MESH_FILE);
    auto Cells = mesh.getCells();
    auto Faces = mesh.getFaces();
    CC.N_MESH_CELL = Cells.size();
    mesh.SaveMeshInfo("output/3D/log/mesh_info.txt");

    Eigen::VectorXd Vi = Eigen::VectorXd::Zero(3);
    Vi << 1.0, 1.0, 1.0;
    std::vector<int> comp_odr = mesh.setupComputationOrder(Vi);

    FiniteElement::Integral<DIM> Int(mesh, CC.QUAD_MODE);
    Int.save("output/3D/log");
    Eigen::MatrixXd IntMat = Int.getIntMat();
    alignedVec<Eigen::MatrixXd> MassMat = Int.getMassMat();
    alignedVec<alignedVec<Eigen::MatrixXd>> StfMat = Int.getStiffMat();
    alignedVec<alignedVec<Eigen::MatrixXd>> MassFaceMat = Int.getMassFaceMat();
    alignedVec<alignedVec<Eigen::MatrixXd>> FluxMat = Int.getFluxMat();
    alignedVec<alignedVec<Eigen::VectorXd>> IntFaceMat = Int.getIntFaceMat();
    alignedVec<alignedVec<alignedVec<Eigen::MatrixXd>>> StiffFaceMat = Int.getStiffFaceMat();
    alignedVec<alignedVec<Eigen::MatrixXd>> DivDivMat = Int.getDivDivMat();
    alignedVec<alignedVec<alignedVec<Eigen::MatrixXd>>> StiffFluxMat = Int.getStiffFluxMat();
    
    auto solver = DGSolver::DGSolver_Advection<DIM>(MassMat, StfMat, MassFaceMat, FluxMat, Cells, Faces, comp_odr);    

    int step = 1;
    double t = 0.0;
    double dt = solver.get_dt();
    double res = 10000;
    double sol_norm = 0.0;
    alignedVec<VectorXd> SolutionPrev =  alignedVec<VectorXd>(CC.N_MESH_CELL, VectorXd::Zero(CC.DOF));
    clock_t start, end;
    start = clock();
    
    // Open file to save step and res data
    ofstream step_res_file("output/3D/log/AdvDiff_step_resisual.txt");
    if (!step_res_file.is_open()) {
        cerr << "Error: Cannot open step_res.txt for writing" << endl;
        return 1;
    }

    cout << "********** ********** **** Iteration **** ********** **********" << endl;
    while(step <= 100){
        SolutionPrev = solver.getSolution();
        t += dt;
        Int.updateFlux(mesh, dt);
        solver.solve(Faces, Cells, comp_odr, MassMat, MassFaceMat, FluxMat);
        sol_norm = 0.0;
        for(int i = 0; i < CC.N_MESH_CELL; i++){
            sol_norm += solver.getSolution()[i].norm();
        }
        
        cout << "Step " << step << ";" << "t = "<< t << endl;
        cout << sol_norm << endl;
        // Save step and res to file
        step_res_file << step << " " << t << " " << sol_norm << endl;
        step++;
    }

    end = clock();
    double times;
    times = (double)(end - start)/CLOCKS_PER_SEC;
    cout << "Time: " << times << endl;

    solver.save(Cells, "output/3D/result/Advection_solution_t01_x05.txt", 0, 0.5);
    // Close the step_res file
    step_res_file.close();

    return 0;
}