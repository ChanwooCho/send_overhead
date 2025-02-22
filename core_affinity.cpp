#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <pthread.h>  // Needed for pthread functions
#include <sched.h>    // Needed for CPU affinity

#define ROWS 5120    // Number of rows in matrix A
#define COLS 5120    // Number of columns in matrix A and number of rows in matrix B
#define B_COLS 128   // Number of columns in matrix B

int main() {
    // Allocate memory for matrix A, matrix B, and result matrix C.
    float* A = new float[ROWS * COLS];     // Matrix A: 5120 x 5120
    float* B = new float[COLS * B_COLS];     // Matrix B: 5120 x 128
    float* C = new float[ROWS * B_COLS];     // Result matrix C: 5120 x 128

    // Initialize A and B with random values between 0 and 1.
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i * COLS + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            B[i * B_COLS + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }

    // Set the number of threads to 4.
    omp_set_num_threads(4);

    // Begin the parallel region.
    #pragma omp parallel num_threads(4)
    {
        int thread_id = omp_get_thread_num();
        // Map each thread to a specific core:
        // thread 0 -> core 4, thread 1 -> core 5, etc.
        int core_id = thread_id + 4;

        // Prepare a CPU set and add the chosen core.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);

        // Get the current thread and set its affinity.
        pthread_t current_thread = pthread_self();
        int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Error setting affinity for thread " 
                      << thread_id << std::endl;
        } else {
            std::cout << "Thread " << thread_id 
                      << " is set to core " << core_id << std::endl;
        }

        // Record start time for this thread.
        double start_time = omp_get_wtime();

        // Each thread computes a part of the matrix multiplication.
        int duty = ROWS / 4;
        int start = thread_id * duty;
        int end = (thread_id + 1) * duty;
        printf("Thread %d: start = %d, end = %d\n", thread_id, start, end);

        for (int i = start; i < end; i++) {
            for (int j = 0; j < B_COLS; j++) {
                float sum = 0.0f;
                for (int k = 0; k < COLS; k++) {
                    sum += A[i * COLS + k] * B[k * B_COLS + j];
                }
                C[i * B_COLS + j] = sum;
            }
        }

        // Record the end time and print the execution time.
        double end_time = omp_get_wtime();
        double thread_time = end_time - start_time;
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id 
                      << " execution time: " << thread_time 
                      << " seconds." << std::endl;
        }
    }

    // Print the first 10 elements of the result matrix.
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
