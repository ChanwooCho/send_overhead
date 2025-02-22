#include <iostream>
#include <omp.h>
#include <cstdlib>  // For rand and srand
#include <ctime>    // For time

#define N 4096  // Matrix size

int main() {
    // Allocate memory for matrices A, B, and C as 1D arrays.
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    // Seed the random number generator.
    srand(time(0));

    // Fill matrices A and B with random numbers (0 to 9).
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = 0;  // Initialize C with zeros.
    }

    // Set the number of threads to 4.
    omp_set_num_threads(4);

    // The outer loop is divided among 4 threads.
    // 'schedule(static, N/4)' assigns N/4 rows per thread.
    #pragma omp parallel for schedule(static, N/4)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    // For testing, print the first 10x10 block of matrix C.
    std::cout << "First 10x10 block of result matrix C:" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free the allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
