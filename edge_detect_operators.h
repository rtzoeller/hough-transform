#ifndef HOUGH_EDGE_DETECT_OPERATORS_H
#define HOUGH_EDGE_DETECT_OPERATORS_H

#include <armadillo>
#include <memory>

template<typename T>
std::unique_ptr<arma::Mat<T>> get_sobel_x() {
    std::unique_ptr<arma::Mat<T>> Gx = std::make_unique<arma::Mat<T>>(3, 3);
    *Gx = {{-1, 0, 1},
           {-2, 0, 2},
           {-1, 0, 1}};
    return Gx;
}

template<typename T>
std::unique_ptr<arma::Mat<T>> get_sobel_y() {
    std::unique_ptr<arma::Mat<T>> Gy = std::make_unique<arma::Mat<T>>(3, 3);
    *Gy = {{-1, -2, -1},
           {0,  0,  0},
           {1,  2,  1}};
    return Gy;
}

template<typename T>
std::unique_ptr<arma::Mat<T>> get_scharr_x() {
    std::unique_ptr<arma::Mat<T>> Gx = std::make_unique<arma::Mat<T>>(3, 3);
    *Gx = {{-3,  0, 3},
           {-10, 0, 10},
           {-3,  0, 3}};
    return Gx;
}

template<typename T>
std::unique_ptr<arma::Mat<T>> get_scharr_y() {
    std::unique_ptr<arma::Mat<T>> Gy = std::make_unique<arma::Mat<T>>(3, 3);
    *Gy = {{-3, -10, -3},
           {0,  0,   0},
           {3,  10,  3}};
    return Gy;
}

template<typename T>
std::unique_ptr<arma::Mat<T>> get_prewitt_x() {
    std::unique_ptr<arma::Mat<T>> Gx = std::make_unique<arma::Mat<T>>(3, 3);
    *Gx = {{-1, 0, 1},
           {-1, 0, 1},
           {-1, 0, 1}};
    return Gx;
}

template<typename T>
std::unique_ptr<arma::Mat<T>> get_prewitt_y() {
    std::unique_ptr<arma::Mat<T>> Gy = std::make_unique<arma::Mat<T>>(3, 3);
    *Gy = {{-1, -1, -1},
           {0,  0,  0},
           {1,  1,  1}};
    return Gy;
}

#endif //HOUGH_EDGE_DETECT_OPERATORS_H
