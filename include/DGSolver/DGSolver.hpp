#pragma once

#include <Eigen/Dense>
#include <omp.h>
#include "GlobalConfig/GlobalConfig.hpp"
#include "SpatialMesh/SpatialMesh.hpp"
#include "FiniteElement/Integral.hpp"
#include "Utility/math_utils.hpp"

using namespace Eigen;
using namespace std;

namespace DGSolver {

    template<int Dim>
    class DGSolver {
        public:
            DGSolver() = default;
            virtual ~DGSolver() = default;
    };

} // namespace DGSolver