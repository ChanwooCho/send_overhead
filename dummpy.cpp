#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>


#define ROWS 5120   // Number of rows in matrix A
#define COLS 5120   // Number of columns in matrix A and number of rows in vector B

int main() {
    // Allocate memory for matrix A, vector B, and result vector C.
    float* A = new float[ROWS * COLS];  // Matrix A: 5120 x 5120
    float* B = new float[COLS];           // Vector B: 5120 x 1
    float* C = new float[ROWS];           // Result vector C: 5120 x 1

    // Initialize A and B with random values between 0 and 1.
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i * COLS + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for (int j = 0; j < COLS; j++) {
        B[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Set the number of threads to 4.
    omp_set_num_threads(4);

    // Perform the matrix multiplication: C = A * B.
    // Each thread gets a chunk of rows to process.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ROWS; i++) {
        float sum = 0.0f;
        for (int j = 0; j < COLS; j++) {
            sum += A[i * COLS + j] * B[j];
        }
        C[i] = sum;
    }

    // Print the first 10 elements of the result vector for a quick check.
    std::cout << "First 10 results:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
