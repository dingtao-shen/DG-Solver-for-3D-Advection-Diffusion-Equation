#pragma once

#include <Eigen/Dense>

namespace BCIC {

    // x = 0
    inline Eigen::VectorXd Bx0(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (1.0 + exp(-Pts(1, i)) + exp(-Pts(2, i))) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bx0dx(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = 0;
        }
        return Val;
    }

    inline Eigen::VectorXd Bx0dy(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(1, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bx0dz(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(2, i)) * exp(t);
        }
        return Val;
    }

    // x = 1
    inline Eigen::VectorXd Bx1(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (exp(-1.0) + exp(-Pts(1, i)) + exp(-Pts(2, i))) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bx1dx(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = 0;
        }
        return Val;
    }

    inline Eigen::VectorXd Bx1dy(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(1, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bx1dz(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(2, i)) * exp(t);
        }
        return Val;
    }

    // y = 0
    inline Eigen::VectorXd By0(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (exp(-Pts(0, i)) + 1.0 + exp(-Pts(2, i))) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd By0dx(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(0, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd By0dy(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = 0;
        }
        return Val;
    }

    inline Eigen::VectorXd By0dz(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(2, i)) * exp(t);
        }
        return Val;
    }

    // y = 1
    inline Eigen::VectorXd By1(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (exp(-Pts(0, i)) + exp(-1.0) + exp(-Pts(2, i))) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd By1dx(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(0, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd By1dy(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = 0;
        }
        return Val;
    }

    inline Eigen::VectorXd By1dz(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(2, i)) * exp(t);
        }
        return Val;
    }

    // z = 0
    inline Eigen::VectorXd Bz0(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (exp(-Pts(0, i)) + exp(-Pts(1, i)) + 1.0) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bz0dx(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(0, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bz0dy(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(1, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bz0dz(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = 0;
        }
        return Val;
    }

    // z = 1
    inline Eigen::VectorXd Bz1(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (exp(-Pts(0, i)) + exp(-Pts(1, i)) + exp(-1.0)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bz1dx(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(0, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bz1dy(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = -exp(-Pts(1, i)) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd Bz1dz(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = 0;
        }
        return Val;
    }
    // analytical solution
    inline Eigen::VectorXd Analytical(Eigen::MatrixXd& Pts, double t){
        Eigen::VectorXd Val = Eigen::VectorXd::Zero(Pts.cols());
        for(int i = 0; i < Pts.cols(); i++){
            Val(i) = (exp(-Pts(0, i)) + exp(-Pts(1, i)) + exp(-Pts(2, i))) * exp(t);
        }
        return Val;
    }

    inline Eigen::VectorXd EvalF(Eigen::MatrixXd& Pts, double t, int bd_index, int dir){
        if(bd_index == 0){
            return Analytical(Pts, t);
        }
        else if(bd_index == 16){
            if(dir == 0){
                return Bx0(Pts, t);
            }
            else if(dir == 1){
                return Bx0dx(Pts, t);
            }
            else if(dir == 2){
                return Bx0dy(Pts, t);
            }
            else if(dir == 3){
                return Bx0dz(Pts, t);
            }
        }
        else if(bd_index == 15){
            if(dir == 0){
                return Bx1(Pts, t);
            }
            else if(dir == 1){
                return Bx1dx(Pts, t);
            }
            else if(dir == 2){
                return Bx1dy(Pts, t);
            }
            else if(dir == 3){
                return Bx1dz(Pts, t);
            }
        }
        else if(bd_index == 13){
            if(dir == 0){
                return By0(Pts, t);
            }
            else if(dir == 1){
                return By0dx(Pts, t);
            }
            else if(dir == 2){
                return By0dy(Pts, t);
            }
            else if(dir == 3){
                return By0dz(Pts, t);
            }
        }
        else if(bd_index == 14){
            if(dir == 0){
                return By1(Pts, t);
            }
            else if(dir == 1){
                return By1dx(Pts, t);
            }
            else if(dir == 2){
                return By1dy(Pts, t);
            }
            else if(dir == 3){
                return By1dz(Pts, t);
            }
        }
        else if(bd_index == 18){
            if(dir == 0){
                return Bz0(Pts, t);
            }
            else if(dir == 1){
                return Bz0dx(Pts, t);
            }
            else if(dir == 2){
                return Bz0dy(Pts, t);
            }
            else if(dir == 3){
                return Bz0dz(Pts, t);
            }
        }
        else if(bd_index == 17){
            if(dir == 0){
                return Bz1(Pts, t);
            }
            else if(dir == 1){
                return Bz1dx(Pts, t);
            }
            else if(dir == 2){
                return Bz1dy(Pts, t);
            }
            else if(dir == 3){
                return Bz1dz(Pts, t);
            }
        }
        else{
            throw std::runtime_error("Boundary condition index is not valid");
        }
        return Eigen::VectorXd::Zero(Pts.cols());
    }

    // inline Eigen::VectorXd EvalF(Eigen::MatrixXd& Pts, double t, int bd_index, int dir){
    //     if(bd_index == 0){
    //         return Analytical(Pts, t);
    //     }
    //     else if(bd_index == 1){
    //         if(dir == 0){
    //             return Bx0(Pts, t);
    //         }
    //         else if(dir == 1){
    //             return Bx0dx(Pts, t);
    //         }
    //         else if(dir == 2){
    //             return Bx0dy(Pts, t);
    //         }
    //         else if(dir == 3){
    //             return Bx0dz(Pts, t);
    //         }
    //     }
    //     else if(bd_index == 2){
    //         if(dir == 0){
    //             return Bx1(Pts, t);
    //         }
    //         else if(dir == 1){
    //             return Bx1dx(Pts, t);
    //         }
    //         else if(dir == 2){
    //             return Bx1dy(Pts, t);
    //         }
    //         else if(dir == 3){
    //             return Bx1dz(Pts, t);
    //         }
    //     }
    //     else if(bd_index == 3){
    //         if(dir == 0){
    //             return By0(Pts, t);
    //         }
    //         else if(dir == 1){
    //             return By0dx(Pts, t);
    //         }
    //         else if(dir == 2){
    //             return By0dy(Pts, t);
    //         }
    //         else if(dir == 3){
    //             return By0dz(Pts, t);
    //         }
    //     }
    //     else if(bd_index == 4){
    //         if(dir == 0){
    //             return By1(Pts, t);
    //         }
    //         else if(dir == 1){
    //             return By1dx(Pts, t);
    //         }
    //         else if(dir == 2){
    //             return By1dy(Pts, t);
    //         }
    //         else if(dir == 3){
    //             return By1dz(Pts, t);
    //         }
    //     }
    //     else if(bd_index == 5){
    //         if(dir == 0){
    //             return Bz0(Pts, t);
    //         }
    //         else if(dir == 1){
    //             return Bz0dx(Pts, t);
    //         }
    //         else if(dir == 2){
    //             return Bz0dy(Pts, t);
    //         }
    //         else if(dir == 3){
    //             return Bz0dz(Pts, t);
    //         }
    //     }
    //     else if(bd_index == 6){
    //         if(dir == 0){
    //             return Bz1(Pts, t);
    //         }
    //         else if(dir == 1){
    //             return Bz1dx(Pts, t);
    //         }
    //         else if(dir == 2){
    //             return Bz1dy(Pts, t);
    //         }
    //         else if(dir == 3){
    //             return Bz1dz(Pts, t);
    //         }
    //     }
    //     else{
    //         throw std::runtime_error("Boundary condition index is not valid");
    //     }
    //     return Eigen::VectorXd::Zero(Pts.cols());
    // }
}