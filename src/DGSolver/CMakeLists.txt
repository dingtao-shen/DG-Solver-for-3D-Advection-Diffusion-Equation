# Create DGSolver library
add_library(DGSolver
    DGSolver_Advection.cpp
)

# Link libraries
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(DGSolver
        PRIVATE
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        Utility
    )
endif()

# Include directories
target_include_directories(DGSolver
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
) 