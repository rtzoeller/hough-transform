#include <iostream>
#include <cstring>
#include "utility.h"

void argparse(int argc, char **argv, int &num_threads, int &threshold, bool &verbose, std::string &output_file,
              std::vector<std::string> &input_files) {
    bool input_file_set = false;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') {
            // This is a flag
            if (strcmp(argv[i], "-j") == 0) {
                // Number of threads
                if (i + 1 == argc) {
                    std::cout << "Unspecified number of threads." << std::endl;
                    std::exit(-1);
                }
                std::string s(argv[++i]);
                num_threads = std::stoi(s);
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 == argc) {
                    std::cout << "Unspecified threshold." << std::endl;
                    std::exit(-1);
                }
                std::string s(argv[++i]);
                threshold = std::stoi(s);
            } else if (strcmp(argv[i], "-o") == 0) {
                if (i + 1 == argc) {
                    std::cout << "Unspecified output file." << std::endl;
                    std::exit(-1);
                }
                output_file = argv[++i];
            } else if (strcmp(argv[i], "-v") == 0) {
                verbose = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                std::cout << "Detects the largest line in an image.\n\n"
                        "Example usage: `./Hough -j 8 input_file.png` -- computes the Hough transform on input_file.png"
                        " using 8 threads and stores the result to output.png (the default).\n\n"
                        "Flags:\n"
                        "\t-h\tDisplays this help message.\n"
                        "\t-j\tNumber of threads to run on.\n"
                        "\t-t\tThreshold to use when scanning edge matrix.\n"
                        "\t-o\tOutput file. Defaults to `output.png`."
                        " If multiple input files are given, this option is ignored and the output files will take the"
                        " form `output#.png`, where # is their position in the argument list.\n"
                        "\t-v\tSave the accumulation matrix as a heatmap." << std::endl;
                std::exit(0);

            } else {
                std::cout << "Unknown parameter " << argv[i] << "." << std::endl;
                std::exit(-1);
            }
        } else {
            // This is a filename
            input_files.push_back(argv[i]);
            input_file_set = true;
        }
    }

    if (!input_file_set) {
        std::cout << "Need to specify an input file. Use `-h` for help." << std::endl;
        std::exit(-1);
    }

}

void print_timestamped(std::string message, std::chrono::steady_clock::time_point time_zero) {
    std::chrono::steady_clock::time_point current = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(current - time_zero).count() << ":\t" <<
    message << std::endl;
}
