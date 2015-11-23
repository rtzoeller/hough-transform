#include <armadillo>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include "edge_detect_operators.h"


template<typename T>
std::unique_ptr<arma::Mat<T>> convolve(arma::Mat<T> &matrix, arma::Mat<T> &kernel) {
    // TODO: Specialize for separable kernels
    unsigned long long Mm, Mn, Km, Kn, Kcx, Kcy, mm, nn, ii, jj;
    Mm = arma::size(matrix).n_rows;
    Mn = arma::size(matrix).n_cols;
    Km = arma::size(kernel).n_rows;
    Kn = arma::size(kernel).n_cols;
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
                        sum += matrix(ii, jj) * kernel(mm, nn);
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
void save_image_grayscale(arma::Mat<T> &matrix, std::string filename) {
    // Scale down to a 0-255 range

    arma::Mat<T> scaled(matrix);
    scaled *= 255;
    scaled /= matrix.max();
    arma::Mat<uint8_t> bytes = arma::conv_to<arma::Mat<uint8_t>>::from(scaled);
    bytes = bytes.t();
    cv::Mat image(arma::size(bytes).n_cols, arma::size(bytes).n_rows, CV_8UC1, bytes.memptr());
    cv::imwrite(filename, image);
}

void print_timestamped(std::string message, std::chrono::steady_clock::time_point time_zero) {
    std::chrono::steady_clock::time_point current = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(current - time_zero).count() << ":\t" << message << std::endl;
}

int main(int argc, char **argv) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const int theta_max = 1000;
    const double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148;

    if (argc == 1) {
        std::cout << "Please give a threshold value." << std::endl;
        return -1;
    }
    std::string threshold_s(argv[1]);
    int threshold = std::stoi(threshold_s);

    // Read in the image and convert it to an Armadillo matrix
    std::unique_ptr<arma::Mat<int>> image;
    try {
        image = open_image_grayscale<int>("test2.png");
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
    arma::Mat<int> edge = arma::sqrt<arma::Mat<int>>(
            arma::square<arma::Mat<int>>(*edge_x) + arma::square<arma::Mat<int>>(*edge_y));

    // Cheat until the convolution handles edges properly (we zero them here)
    for (unsigned long long i = 0; i < arma::size(edge).n_rows; i++) {
        edge(i, 0) = 0;
        edge(i, arma::size(edge).n_cols - 1) = 0;
    }
    for (unsigned long long i = 0; i < arma::size(edge).n_cols; i++) {
        edge(0, i) = 0;
        edge(arma::size(edge).n_rows - 1, i) = 0;
    }
    print_timestamped("Successfully generated edge matrix.", start);

    // Go through every nonzero element and draw lines through it in polar normal form, accumulating the result in a seperate matrix
    const long long rho_max = std::llround(std::sqrt(arma::size(edge).n_rows * arma::size(edge).n_rows + arma::size(edge).n_cols * arma::size(edge).n_cols));
    arma::Mat<int> acc(1 + 2 * theta_max, 1 + 2 * rho_max, arma::fill::zeros);
    arma::uvec nonzero = arma::find(edge > threshold);
    for (unsigned long long i = 0; i < arma::size(nonzero).n_rows; i++) {
        int x = nonzero(i) / arma::size(edge).n_rows;
        int y = nonzero(i) % arma::size(edge).n_rows;

        for (unsigned long long j = 0; j < arma::size(acc).n_rows; j++) {
            double theta = pi * ((((double) j) - theta_max) / (2 * theta_max));
            long long rho_rounded = std::llround((x * std::cos(theta) + y * std::sin(theta)));
            long long k = rho_rounded + rho_max;
            acc(j, k) = acc(j, k) + 1;
        }

    }
    print_timestamped("Successfully generated accumulator matrix.", start);

    save_image_grayscale(acc, "heatmap.png");

    unsigned long long max_row = 0;
    unsigned long long max_col = 0;
    int max = acc.max(max_row, max_col);

    double theta = pi * ((((double) max_row) - theta_max) / (2 * theta_max));
    long long rho = max_col - rho_max;

    double x = rho * std::cos(theta);
    double y = rho * std::sin(theta);
    double slope = -y / x;
    long long right_side_x = arma::size(*image).n_cols - 1;
    long long left_inter = std::round(slope * (0 - x) + y);
    long long right_inter = std::round(slope * (right_side_x - x) + y);

    cv::Mat lined = cv::imread("test2.png", CV_LOAD_IMAGE_GRAYSCALE);
    cv::line(lined, cv::Point(0, left_inter), cv::Point(right_side_x, right_inter), cv::Scalar(33, 33, 33));
    cv::imwrite("output.png", lined);

    std::cout << "Max: " << max << " at (theta, rho): " << theta << ", " << rho << std::endl;

    return 0;
}