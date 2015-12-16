#include <armadillo>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include "edge_detection.h"
#include "hough_transform.h"
#include "image_io.h"

void print_timestamped(std::string message, std::chrono::steady_clock::time_point time_zero) {
    std::chrono::steady_clock::time_point current = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(current - time_zero).count() << ":\t" << message << std::endl;
}

int main(int argc, char **argv) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    int num_threads = 1;
    int threshold = 100;
    std::string output_file = "output.png";
    bool input_file_set = false;
    std::string input_file;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            // This is a flag
            if (strcmp(argv[i], "-j") == 0) {
                // Number of threads
                if (i + 1 == argc) {
                    std::cout << "Unspecified number of threads." << std::endl;
                    return -1;
                }
                std::string s(argv[++i]);
                num_threads = std::stoi(s);
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 == argc) {
                    std::cout << "Unspecified threshold." << std::endl;
                    return -1;
                }
                std::string s(argv[++i]);
                threshold = std::stoi(s);
            } else if (strcmp(argv[i], "-o") == 0) {
                if (i + 1 == argc) {
                    std::cout << "Unspecified output file." << std::endl;
                    return -1;
                }
                output_file = argv[++i];
            } else if (strcmp(argv[i], "-h") == 0) {
                std::cout << "Detects the largest line in an image.\n\n"
                        "Example usage: `./Hough -j 8 input_file.png` -- computes the Hough transform on input_file.png"
                        " using 8 threads and stores the result to output.png.\n\n"
                        "Flags:\n"
                        "\t-h\tDisplays this help message.\n"
                        "\t-j\tNumber of threads to run on.\n"
                        "\t-t\tThreshold to use when scanning edge matrix.\n"
                        "\t-o\tOutput file (defaults to output.png).\n" << std::endl;
                return 0;

            } else {
                std::cout << "Unknown parameter " << argv[i] << std::endl;
                return -1;
            }
        } else {
            // This is the filename
            if (input_file_set) {
                std::cout << "Arguments may only include one non-identified parameter - the file name." << std::endl;
                return -1;
            }
            input_file = argv[i];
            input_file_set = true;
        }
    }

    if (!input_file_set) {
        std::cout << "Need to specify an input file. Use `-h` for help." << std::endl;
        return -1;
    }

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

    // Compute the Hough transformation accumulation matrix
    std::unique_ptr<arma::Mat<int>> acc = hough(edge, threshold, num_threads);
    print_timestamped("Successfully generated accumulator matrix.", start);

    save_image_grayscale(*acc, "heatmap.png");

    unsigned long long max_row = 0;
    unsigned long long max_col = 0;
    acc->max(max_row, max_col);

    const long long rho_max = std::llround(std::sqrt(edge.n_rows * edge.n_rows + edge.n_cols * edge.n_cols));
    const double theta = pi * ((((double) max_row) - theta_max) / (2 * theta_max));
    const long long rho = max_col - rho_max;
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

    cv::imwrite(output_file, lined);
    print_timestamped("Successfully saved output image.", start);
    return 0;
}