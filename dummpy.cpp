#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define ROWS 5120    // Number of rows in matrix A
#define COLS 5120    // Number of columns in matrix A and number of rows in matrix B
#define B_COLS 128   // Number of columns in matrix B

int main() {
    // Allocate memory for matrix A, matrix B, and result matrix C.
    float* A = new float[ROWS * COLS];     // Matrix A: 5120 x 5120
    float* B = new float[COLS * B_COLS];     // Matrix B: 5120 x 128
    float* C = new float[ROWS * B_COLS];     // Result matrix C: 5120 x 128

    // Initialize A with random values between 0 and 1.
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i * COLS + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Initialize B with random values between 0 and 1.
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            B[i * B_COLS + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Set the number of threads to 4.
    omp_set_num_threads(4);

    // Perform the matrix multiplication: C = A * B.
    // C[i][j] = sum( A[i][k] * B[k][j] ) for k = 0 to COLS-1.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            float sum = 0.0f;
            for (int k = 0; k < COLS; k++) {
                sum += A[i * COLS + k] * B[k * B_COLS + j];
            }
            C[i * B_COLS + j] = sum;
        }
    }

    // Print the first 10 elements of the result matrix for a quick check.
    std::cout << "First 10 results:" << std::endl;
    for (int i = 0; i < 10 && i < ROWS * B_COLS; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free the allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
