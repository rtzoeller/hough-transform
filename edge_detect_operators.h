#ifndef HOUGH_EDGE_DETECT_OPERATORS_H
#define HOUGH_EDGE_DETECT_OPERATORS_H

#include <armadillo>
#include <memory>

template<typename T>
std::unique_ptr<arma::Mat<T>> get_sobel_x() {
    std::unique_ptr<arma::Mat<T>> Gx = std::make_unique<arma::Mat<T>>(3, 3);
    *Gx << -1 << 0 << 1 << arma::endr
    << -2 << 0 << 2 << arma::endr
    << -1 << 0 << 1 << arma::endr;
    return Gx;
}

template<typename T>
std::unique_ptr<arma::Mat<T>> get_sobel_y() {
    std::unique_ptr<arma::Mat<T>> Gy = std::make_unique<arma::Mat<T>>(3, 3);
    *Gy << -1 << -2 << -1 << arma::endr
    << 0 << 0 << 0 << arma::endr
    << 1 << 2 << 1 << arma::endr;
    return Gy;
}

#endif //HOUGH_EDGE_DETECT_OPERATORS_H
