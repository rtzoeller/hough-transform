#ifndef HOUGH_UTILITY_H
#define HOUGH_UTILITY_H

#include <chrono>
#include <vector>

/**
 * \file utility.h Utility functions related not directly related to the Hough transform
 */

/**
 * \brief Parse command line arguments
 *
 * Accepts as arguments the default values for options, which will be overwritten
 * as given in the command line arguments. Calls std::exit on an error.
 *
 * \param[in]       argc        Number of arguments in \p argv
 * \param[in]       argv        Command line arguments
 * \param[in,out]   num_threads Number of threads to distribute the computation to
 * \param[in,out]   threshold   Threshold value for considering a pixel to be an edge
 * \param[in,out]   verbose     Should the program run in verbose mode?
 * \param[in,out]   output_file Filename to save the output image to
 * \param[in,out]   input_files Files to use as input for the Hough transform
 */
void argparse(int argc, char **argv, int &num_threads, int &threshold, bool &verbose, std::string &output_file,
              std::vector<std::string> &input_files);

/**
 * \brief Print a timestamped message to stdout
 *
 * \param[in]   message     Message to print
 * \param[in]   time_zero   Reference time to offset the current time by
 */
void print_timestamped(std::string message, std::chrono::steady_clock::time_point time_zero);

#endif //HOUGH_UTILITY_H
