#include <armadillo>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include "edge_detect_operators.h"

const unsigned int num_threads = 8; // Currently hard coded to 8 to match test machine
const int theta_max = 1000;
const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148;

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

template<typename T>
std::unique_ptr<arma::Mat<T>> open_image_grayscale(std::string filename) {
    // Armadillo stores matrices in column major order, OpenCV in row major order,
    // so we have to transpose the matrix before handing it to Armadillo.
    cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    image = image.t();
    arma::Mat<uint8_t> output(image.ptr(), image.size().width, image.size().height);
    return std::make_unique<arma::Mat<T>>(arma::conv_to<arma::Mat<T>>::from(output));
}

template<typename T>
void save_image_grayscale(const arma::Mat<T> &matrix, std::string filename) {
    // Scale down to a 0-255 range

    arma::Mat<T> scaled(matrix);
    scaled *= 255;
    scaled /= matrix.max();
    arma::Mat<uint8_t> bytes = arma::conv_to<arma::Mat<uint8_t>>::from(scaled);
    bytes = bytes.t();
    cv::Mat image(bytes.n_cols, bytes.n_rows, CV_8UC1, bytes.memptr());
    cv::imwrite(filename, image);
}

void print_timestamped(std::string message, std::chrono::steady_clock::time_point time_zero) {
    std::chrono::steady_clock::time_point current = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(current - time_zero).count() << ":\t" << message << std::endl;
}

int main(int argc, char **argv) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    if (argc < 3) {
        std::cout << "Please give a threshold value and filename." << std::endl;
        return -1;
    }
    std::string threshold_s(argv[1]);
    int threshold = std::stoi(threshold_s);
    std::string input_file(argv[2]);

    // Read in the image and convert it to an Armadillo matrix
    std::unique_ptr<arma::Mat<int>> image;
    try {
        image = open_image_grayscale<int>(input_file);
    } catch (cv::Exception e) {
        std::cout << "Unable to open file." << std::endl;
        return -1;
    }
    print_timestamped("Successfully converted image.", start);

    // Edge detect operators
    std::unique_ptr<arma::Mat<int>> Gx = get_sobel_x<int>();
    std::unique_ptr<arma::Mat<int>> Gy = get_sobel_y<int>();

    // Apply the edge detect operators
    std::unique_ptr<arma::Mat<int>> edge_x = convolve<int>(*image, *Gx);
    std::unique_ptr<arma::Mat<int>> edge_y = convolve<int>(*image, *Gy);
    arma::Mat<int> edge = arma::sqrt<arma::Mat<int>>(arma::square<arma::Mat<int>>(*edge_x) + arma::square<arma::Mat<int>>(*edge_y));
    print_timestamped("Successfully generated edge matrix.", start);

    // Distribute the work to as many threads as allowed
    std::vector<std::thread> pool(num_threads);

    const long long rho_max = std::llround(std::sqrt(edge.n_rows * edge.n_rows + edge.n_cols * edge.n_cols));
    arma::uvec nonzero = arma::find(edge > threshold);
    arma::Mat<int> acc(1 + 2 * theta_max, 1 + 2 * rho_max, arma::fill::zeros);
    for (unsigned int i = 0; i < num_threads; i++) {
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
        pool.at(i) = std::thread(hough_accumulate, lower, upper, std::ref(acc), std::ref(nonzero), std::ref(edge));
    }
    // Merge the worker threads back into the main one
    for (unsigned int i = 0; i < num_threads; i++) {
        pool.at(i).join();
    }
    print_timestamped("Successfully generated accumulator matrix.", start);

    save_image_grayscale(acc, "heatmap.png");

    unsigned long long max_row = 0;
    unsigned long long max_col = 0;
    acc.max(max_row, max_col);

    double theta = pi * ((((double) max_row) - theta_max) / (2 * theta_max));
    long long rho = max_col - rho_max;
    cv::Mat lined = cv::imread(input_file, CV_LOAD_IMAGE_GRAYSCALE);

    if (theta == 0) {
        cv::line(lined, cv::Point(rho, 0), cv::Point(rho, (*image).n_rows - 1), cv::Scalar(33, 33, 33));
    } else {
        double x = rho * std::cos(theta);
        double y = rho * std::sin(theta);
        double slope = -x / y;
        long long right_side_x = (*image).n_cols - 1;
        long long left_inter = std::llround(slope * (0 - x) + y);
        long long right_inter = std::llround(slope * (right_side_x - x) + y);
        cv::line(lined, cv::Point(0, left_inter), cv::Point(right_side_x, right_inter), cv::Scalar(33, 33, 33));
    }

    cv::imwrite("output.png", lined);
    print_timestamped("Successfully saved output image.", start);
    return 0;
}