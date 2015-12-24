#include "hough_transform.h"
#include <thread>

void hough_accumulate(unsigned int lower, unsigned int upper, arma::Mat<int> &acc, const arma::uvec &nonzero,
                      const arma::Mat<int> &edge) {
    // Go through every nonzero element and draw lines through it in polar normal form, accumulating the result in a seperate matrix
    const long long rho_max = std::llround(std::sqrt(edge.n_rows * edge.n_rows + edge.n_cols * edge.n_cols));
    for (unsigned long long j = lower; j < upper; j++) {
        double theta = pi * ((((double) j) - theta_max) / (2 * theta_max));
        double s = std::sin(theta);
        double c = std::cos(theta);

        for (unsigned long long i = 0; i < nonzero.n_rows; i++) {
            int x = nonzero(i) / edge.n_rows;
            int y = nonzero(i) % edge.n_rows;
            long long rho_rounded = std::llround((x * c + y * s));
            long long k = rho_rounded + rho_max;
            acc(j, k) = acc(j, k) + 1;
        }
    }
}

std::unique_ptr<arma::Mat<int>> hough(const arma::Mat<int> &edge, int threshold, int num_threads) {
    const long long rho_max = std::llround(std::sqrt(edge.n_rows * edge.n_rows + edge.n_cols * edge.n_cols));
    arma::uvec nonzero = arma::find(edge > threshold);
    std::unique_ptr<arma::Mat<int>> acc = std::make_unique<arma::Mat<int>>(1 + 2 * theta_max, 1 + 2 * rho_max,
                                                                           arma::fill::zeros);
    if (num_threads <= 1) {
        hough_accumulate(0, 2 * theta_max + 1, std::ref(*acc), std::ref(nonzero), std::ref(edge));
    } else {
        // Distribute the work to as many threads as allowed
        std::vector<std::thread> pool(num_threads);

        for (int i = 0; i < num_threads; i++) {
            int lower = i * ((2 * theta_max + 1) / num_threads);
            int upper;
            if (i + 1 == num_threads) {
                upper = 2 * theta_max + 1;
            } else {
                upper = (i + 1) * ((2 * theta_max + 1) / num_threads);
            }
            // We pass in a reference to the accumulator matrix itself here, which poses some issues if we
            // are not careful. All threads will have access to the same matrix in memory, so if any threads
            // are modifying the same portion of the matrix data will become corrupted.
            // We solve this by having each thread exclusively write to a unique block of the matrix,
            // seperated by upper and lower bounds on theta. This needs to be enforced strictly.
            pool.at(i) = std::thread(hough_accumulate, lower, upper, std::ref(*acc), std::cref(nonzero),
                                     std::cref(edge));
        }
        // Merge the worker threads back into the main one
        for (int i = 0; i < num_threads; i++) {
            pool.at(i).join();
        }
    }

    return acc;
}