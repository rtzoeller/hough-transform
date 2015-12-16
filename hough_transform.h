#ifndef HOUGH_HOUGH_TRANSFORM_H
#define HOUGH_HOUGH_TRANSFORM_H

#include <armadillo>
#include <memory>

const int theta_max = 1000;
const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148;

std::unique_ptr<arma::Mat<int>> hough(const arma::Mat<int> &edge, int threshold, int num_threads);

#endif //HOUGH_HOUGH_TRANSFORM_H
