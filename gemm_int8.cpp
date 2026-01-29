
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <pthread.h>
#include <sched.h>      // For sched_setaffinity and sched_getcpu
#include <unistd.h>     // For syscall and sysconf
#include <sys/syscall.h> // For SYS_gettid
#include <cerrno>
#include <cstring>

#define ROWS 5120   // Number of rows in matrix A
#define COLS 5120   // Number of columns in matrix A and number of rows in matrix B
#define B_COLS 1    // Number of columns in matrix B

int main() {
    // Check and print the number of available cores.
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout << "Number of available cores: " << num_cores << std::endl;
    
    // Allocate memory for matrices A, B, and C using int type.
    int* A = new int[ROWS * COLS];
    int* B = new int[COLS * B_COLS];
    int* C = new int[ROWS * B_COLS];

    // Initialize random seed.
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            // Random value between -128 and 127.
            A[i * COLS + j] = static_cast<int8_t>(rand() % 256 - 128);
        }
    }
    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < B_COLS; j++) {
            B[i * B_COLS + j] = static_cast<int8_t>(rand() % 256 - 128);
        }
    }

    // Set the number of OpenMP threads to 4.
    omp_set_num_threads(4);

    // Start parallel region.
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // Map threads: thread 0 -> core 4, thread 1 -> core 5, etc.
        int desired_core = thread_id + 4;

        // Check if the desired core is available.
        if (desired_core >= num_cores) {
            std::cerr << "Warning: Thread " << thread_id 
                      << " requested core " << desired_core 
                      << ", but only " << num_cores 
                      << " cores are available." << std::endl;
        } else {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(desired_core, &cpuset);

            // Get the thread ID.
            pid_t tid = syscall(SYS_gettid);
            // Set the CPU affinity.
            if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
                // Uncomment below to see error messages.
                // std::cerr << "Error setting affinity for thread " << thread_id 
                //           << " (core " << desired_core << "): " 
                //           << std::strerror(errno) << std::endl;
            } else {
                // Uncomment below for success messages.
                // std::cout << "Thread " << thread_id 
                //           << " is set to core " << desired_core << std::endl;
            }
        }

        // For debugging: print the actual CPU on which the thread is running.
        int actual_cpu = sched_getcpu();

        // Record start time.
        double start_time = omp_get_wtime();

        // Determine the work for this thread.
        int duty = ROWS / 4;
        int start = thread_id * duty;
        int end = (thread_id + 1) * duty;

        // Matrix multiplication using int type.
        for (int i = start; i < end; i++) {
            for (int j = 0; j < B_COLS; j++) {
                int sum = 0;
                for (int k = 0; k < COLS; k++) {
                    sum += A[i * COLS + k] * B[k * B_COLS + j];
                }
                C[i * B_COLS + j] = sum;
            }
        }

        // Calculate and print execution time.
        double thread_time = omp_get_wtime() - start_time;
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id 
                      << " execution time: " << thread_time * 1000 * 1000 
                      << " us" << std::endl;
            std::cout << "Thread " << thread_id 
                      << " is actually running on CPU " << actual_cpu << std::endl;
        }
    }

    // Print the first 10 elements of matrix C.
    std::cout << "First 10 results:" << std::endl;
    for (int i = 0; i < 10 && i < ROWS * B_COLS; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Free allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
