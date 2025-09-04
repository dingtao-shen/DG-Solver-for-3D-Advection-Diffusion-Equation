#pragma once

#include <yaml-cpp/yaml.h>
#include <unordered_map>
#include <vector>
#include <string>
#include "Eigen/Dense"

template<typename T>
using alignedVec = std::vector<T, Eigen::aligned_allocator<T>>;

struct ControlConstants {
    int SPATIAL_DIM;
    int POLYDEG;
    int QUAD_MODE;
    int DOF;
    std::string MODEL;
    int TMAX;
    double TOL;

    int MESH_TYPE;
    std::string MESH_FILE;
    int N_BOUNDARY;
    std::unordered_map<int, std::pair<int, double>> BOUNDARY_COND;
    int N_MESH_CELL;

    void LoadFromFile(const std::string& filename){
        YAML::Node config = YAML::LoadFile(filename);

        SPATIAL_DIM = config["SPATIAL_DIM"].as<int>();
        POLYDEG = config["POLYDEG"].as<int>();
        QUAD_MODE = config["QUAD_MODE"].as<int>();
        if(SPATIAL_DIM == 1) {
            DOF = POLYDEG + 1;
        }
        else if(SPATIAL_DIM == 2) {
            DOF = (POLYDEG + 1) * (POLYDEG + 2) / 2;
        }
        else if(SPATIAL_DIM == 3) {
            DOF = (POLYDEG + 1) * (POLYDEG + 2) * (POLYDEG + 3) / 6;
        }

        MODEL = config["MODEL"].as<std::string>();

        TMAX = config["TMAX"].as<int>();
        TOL = config["TOL"].as<double>();

        MESH_TYPE = config["MESH_TYPE"].as<int>();
        MESH_FILE = config["MESH_FILE"].as<std::string>();
        N_BOUNDARY = config["N_BOUNDARY"].as<int>();
        for (const auto& boundary : config["BOUNDARY_COND"]) {
            BOUNDARY_COND[boundary.first.as<int>()] = std::make_pair(boundary.second[0].as<int>(), boundary.second[1].as<double>());
        }
    }
};

extern ControlConstants CC;