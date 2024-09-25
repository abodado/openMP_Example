#include <iostream>
#include <omp.h>
#include <cstdlib> // for std::atoi
#include <vector>

int main(int argc, char* argv[]) {
    const int N = 1000; // Size of the matrices
    std::vector<std::vector<int>> A(N, std::vector<int>(N));
    std::vector<std::vector<int>> B(N, std::vector<int>(N));
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));
    
    // Check if the number of threads is provided as an argument
    int numThreads = 1; // Default to 1 thread
    if (argc > 1) {
        numThreads = std::atoi(argv[1]); // Convert argument to integer
        omp_set_num_threads(numThreads); // Set the number of threads
    } else {
        //std::cerr << "Usage: " << argv[0] << " <number_of_threads>" << std::endl;
		//return 1;
		// Use the default value
		omp_set_num_threads(numThreads);        
    }
	
	omp_set_num_threads(numThreads);	

    // Output the number of threads being used
    std::cout << "Using " << numThreads << " threads for matrix multiplication." << std::endl;

    // Initialize matrices A and B
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = i + j;
            B[i][j] = i - j;
        }
    }

    // Start the timer
    double start_time = omp_get_wtime();

    // Perform matrix multiplication C = A * B
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    // Stop the timer
    double end_time = omp_get_wtime();

    // Calculate elapsed time
    double elapsed_time = end_time - start_time;

    std::cout << "Matrix multiplication completed." << std::endl;
    std::cout << "Elapsed Time: " << elapsed_time << " seconds" << std::endl;

    return 0;
}
