#pragma once

#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>
#include "Eigen/Dense"

namespace FiniteElement{

    class Quad3D{
        private:
            Eigen::MatrixXd points;
            Eigen::VectorXd weights;
        public:
            Quad3D(int N){
                points = Eigen::MatrixXd::Zero(4, N);
                weights = Eigen::VectorXd::Zero(N);
                std::ifstream file("./include/FiniteElement/quad_3D_" + std::to_string(N) + ".txt");
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open quadrature file: quad_3D_" + std::to_string(N) + ".txt");
                }
                for (int i = 0; i < N; i++) {
                    double l1, l2, l3, l4, w;
                    if (!(file >> l1 >> l2 >> l3 >> l4 >> w)) {
                        std::cerr << "Error reading quadrature data at line " << (i + 1) << std::endl;
                        break;
                    }
                    points(0, i) = l1;
                    points(1, i) = l2;
                    points(2, i) = l3;
                    points(3, i) = l4;
                    weights(i) = w / 6.0;
                }
            };
            ~Quad3D(){};

            Eigen::MatrixXd Cords(){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(3, 4);
                for(int i = 1; i < 4; i++){
                    Vertices(i-1,i) = 1.0;
                }
                Eigen::MatrixXd CP = Vertices * points;
                // std::cout << "CP = " << CP.transpose() << std::endl;
                return CP;
            }

            // Vertices: 3 x 4
            Eigen::MatrixXd Cords(Eigen::MatrixXd Vertices){
                Eigen::MatrixXd CP = Vertices * points;
                return CP;
            }

            double Int(Eigen::VectorXd FVal){
                assert(FVal.size() == points.cols());
                double IntVal = 0.0;
                for(int i = 0; i < points.cols(); i++){
                    IntVal += weights(i) * FVal(i);
                }
                return IntVal;
            }

            Eigen::MatrixXd pts() const {return points;}
            Eigen::VectorXd wts() const {return weights;}
    };

    class Quad2D{
        private:
            Eigen::MatrixXd points;
            Eigen::VectorXd weights;
        public:
            Quad2D(int N){
                points = Eigen::MatrixXd::Zero(3, N);
                weights = Eigen::VectorXd::Zero(N);
                std::ifstream file("./include/FiniteElement/quad_2D_" + std::to_string(N) + ".txt");
                if (!file.is_open()) {
                    throw std::runtime_error("Could not open quadrature file: quad_2D_" + std::to_string(N) + ".txt");
                }
                for (int i = 0; i < N; i++) {
                    double l1, l2, l3, w;
                    if (!(file >> l1 >> l2 >> l3 >> w)) {
                        std::cerr << "Error reading quadrature data at line " << (i + 1) << std::endl;
                        break;
                    }
                    points(0, i) = l1;
                    points(1, i) = l2;
                    points(2, i) = l3;
                    weights(i) = w / 2.0;
                }
            };
            ~Quad2D(){};

            Eigen::MatrixXd Cords(){
                Eigen::MatrixXd Vertices = Eigen::MatrixXd::Zero(2, 3);
                for(int i = 1; i < 3; i++){
                    Vertices(i-1,i) = 1.0;
                }
                Eigen::MatrixXd CP = Vertices * points;
                // std::cout << "CP = " << CP.transpose() << std::endl;
                return CP;
            }

            // Vertices: 3 x 3
            Eigen::MatrixXd Cords(Eigen::MatrixXd Vertices){
                assert(Vertices.rows() == 3); // 3D points
                Eigen::MatrixXd CP = Vertices * points;
                return CP;
            }

            double Int(Eigen::VectorXd FVal){
                assert(FVal.size() == points.cols());
                double IntVal = 0.0;
                for(int i = 0; i < points.cols(); i++){
                    IntVal += weights(i) * FVal(i);
                }
                return IntVal;
            }

            Eigen::MatrixXd pts() const {return points;}
            Eigen::VectorXd wts() const {return weights;}
    };

}