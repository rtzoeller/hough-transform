#ifndef HOUGH_HOUGH_TRANSFORM_H
#define HOUGH_HOUGH_TRANSFORM_H

#include <armadillo>
#include <memory>

/**
 * \file hough_transform.h The Hough transform for detecting lines in an image
 */

const int theta_max = 1000;
const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148;

/**
 * \brief Generate the accumulator matrix via the Hough transform
 *
 * \param[in]   edge        The edge matrix to process
 * \param[in]   threshold   The minimum value for a pixel to be considered an edge
 * \param[in]   num_threads The number of threads to distribute the workload across
 *
 * \returns The accumulator matrix produced by the Hough transform
 */
std::unique_ptr<arma::Mat<int>> hough(const arma::Mat<int> &edge, int threshold, int num_threads);

#endif //HOUGH_HOUGH_TRANSFORM_H
