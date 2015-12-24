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

template<typename T>
std::unique_ptr<arma::Mat<T>> convolve(const arma::Mat<T> &matrix, const arma::Mat<T> &kernel) {
    // TODO: Specialize for separable kernels
    unsigned long long Mm, Mn, Km, Kn, Kcx, Kcy, mm, nn, ii, jj;
    Mm = matrix.n_rows;
    Mn = matrix.n_cols;
    Km = kernel.n_rows;
    Kn = kernel.n_cols;
    Kcx = Kn / 2;
    Kcy = Km / 2;

    std::unique_ptr<arma::Mat<T>> result = std::make_unique<arma::Mat<T>>(arma::size(matrix));

    for (unsigned long long i = 0; i < Mm; i++) {
        for (unsigned long long j = 0; j < Mn; j++) {
            T sum = 0;
            for (unsigned long long y = 0; y < Km; y++) {
                mm = Km - 1 - y;
                for (unsigned long long x = 0; x < Kn; x++) {
                    nn = Kn - 1 - x;
                    ii = i + (y - Kcy);
                    jj = j + (x - Kcx);

                    if (ii >= 0 && ii < Mm && jj >= 0 && jj < Mn) {
                        // Normal case, we aren't trying to read from an invalid index
                        sum += matrix(ii, jj) * kernel(mm, nn);

                    } else {
                        // We are trying to read from an invalid index, so we extend the last valid element outward
                        if (ii < 0) {
                            if (jj < 0) {
                                // Top left
                                sum += matrix(0, 0) * kernel(mm, nn);
                            } else if (jj >= Mn) {
                                // Top right
                                sum += matrix(0, Mn - 1) * kernel(mm, nn);
                            } else {
                                // Top middle
                                sum += matrix(0, jj) * kernel(mm, nn);
                            }
                        } else if (ii >= Mm) {
                            if (jj < 0) {
                                // Bottom left
                                sum += matrix(Mm - 1, 0) * kernel(mm, nn);
                            } else if (jj >= Mn) {
                                // Bottom right
                                sum += matrix(Mm - 1, Mn - 1) * kernel(mm, nn);
                            } else {
                                // Bottom middle
                                sum += matrix(Mm - 1, jj) * kernel(mm, nn);
                            }
                        } else if (jj < 0) {
                            if (ii >= 0 && ii < Mm) {
                                // Left middle
                                // We don't handle left top or bottom because those are handled above
                                sum += matrix(ii, 0) * kernel(mm, nn);
                            }
                        } else if (jj >= Mn) {
                            if (ii >= 0 && ii < Mm) {
                                // Right middle
                                // We don't handle right top or bottom because those are handled above
                                sum += matrix(ii, Mm - 1) * kernel(mm, nn);
                            }
                        }

                    }
                }
            }
            (*result)(i, j) = sum;
        }
    }

    return result;
}


#endif //HOUGH_EDGE_DETECT_OPERATORS_H
