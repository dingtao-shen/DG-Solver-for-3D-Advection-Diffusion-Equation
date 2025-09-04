# Add subdirectories for each module
add_subdirectory(Utility)
add_subdirectory(Polynomial)
add_subdirectory(DGSolver)

# # Add main executable
# add_executable(PhononBTE main/PhononBTE.cpp)
# # Include directories
# target_include_directories(PhononBTE
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/../include
#     ${CMAKE_CURRENT_SOURCE_DIR}
# )
# # Link libraries
# target_link_libraries(PhononBTE
#     PRIVATE
#     Utility
#     Eigen3::Eigen
#     OpenMP::OpenMP_CXX
#     yaml-cpp
#     stdc++fs
# )

add_executable(Test test/Test.cpp)
target_include_directories(Test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(Test
    PRIVATE
    Utility
    Polynomial
    DGSolver
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    yaml-cpp
    stdc++fs
)

add_executable(BasisFunctionsTests test/BasisFunctionsTests.cpp)
target_include_directories(BasisFunctionsTests
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(BasisFunctionsTests
    PRIVATE
    Utility
    Polynomial
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    yaml-cpp
    stdc++fs
)