#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define ROWS 5120  // Number of rows in matrix A
#define COLS 5120  // Number of columns in matrix A and rows in matrix B

int main() {
    // Dynamically allocate matrix A (ROWS x COLS) as fp32 (float)
    float** A = new float*[ROWS];
    for (int i = 0; i < ROWS; ++i) {
        A[i] = new float[COLS];
    }
    
    // Allocate matrix B (COLS x 1)
    float* B = new float[COLS];
    
    // Allocate result matrix C (ROWS x 1)
    float* C = new float[ROWS];
    
    // Initialize random seed
    srand(static_cast<unsigned int>(time(NULL)));
    
    // Initialize matrix A with random float numbers in the range [0, 10)
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            A[i][j] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
        }
    }
    
    // Initialize matrix B with random float numbers in the range [0, 10)
    for (int i = 0; i < COLS; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
    
    // Set OpenMP to use 4 threads
    omp_set_num_threads(4);
    
    // Matrix multiplication: C = A * B
    #pragma omp parallel num_threads(4)
    {
        int duty = ROWS / 4;
        int start = duty * omp_get_thread_num();
        int end = duty * (omp_get_thread_num() + 1);
        printf("start = %d, end = %d\n", start, end);
        for (int i = start; i < end; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < COLS; ++j) {
                sum += A[i][j] * B[j];
            }
            C[i] = sum;
        }
    }
    
    // Output the first 10 elements of the result vector
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }
    
    // Free dynamically allocated memory
    for (int i = 0; i < ROWS; ++i) {
        delete [] A[i];
    }
    delete [] A;
    delete [] B;
    delete [] C;
    
    return 0;
}
