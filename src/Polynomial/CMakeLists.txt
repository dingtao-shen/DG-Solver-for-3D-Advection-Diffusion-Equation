# Create Polynomial library
add_library(Polynomial
    polynomial.cpp
)

# Link libraries
target_link_libraries(Polynomial
    PRIVATE
    Utility
    Eigen3::Eigen
)

# Include directories
target_include_directories(Polynomial
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
) 