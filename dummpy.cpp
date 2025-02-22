#include <iostream>
#include <omp.h>
#include <cstdlib>  // for rand and srand
#include <ctime>    // for time

// Set the matrix size
#define N 4096

int main() {
    // Allocate memory for matrices A, B, and C.
    // We use 1D arrays to store 2D data.
    double* A = new double[N * N];
    double* B = new double[N * N];
    double* C = new double[N * N];

    // Seed the random number generator
    srand(time(0));

    // Fill matrices A and B with random numbers (0 to 9).
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
        C[i] = 0;  // Initialize C with zero
    }

    // Multiply matrices: C = A * B.
    // We use three loops. The outer loop is parallelized.
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    // For testing, print the first 10x10 block of the result matrix C.
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
