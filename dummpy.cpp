#include <iostream>
#include <omp.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <sched.h>
#include <pthread.h>

#define SIZE 4096  // Matrix size: 4096 x 4096

// A flag to signal the send threads to stop.
std::atomic<bool> running(true);

// This function simulates a "send" operation.
// It does some dummy work to create an overhead.
void send_function(int id) {
    volatile int dummy = 0;
    // Loop to simulate work (you can adjust the number of iterations)
    for (int i = 0; i < 100000; i++) {
        dummy += (i % 7);
    }
}

// This is the function for each send thread.
// Each thread is pinned to one of cores 0-3 (round-robin assignment).
void send_thread_func(int thread_id) {
    int core_id = thread_id % 4; // Use cores 0, 1, 2, 3
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error setting affinity for send thread " << thread_id << ": " << rc << "\n";
    }
    
    // Keep calling send_function while matrix multiplication is running.
    while (running.load()) {
        send_function(thread_id);
    }
}

// This function performs matrix multiplication of two SIZE x SIZE matrices.
// It uses an OpenMP parallel region where each thread is pinned to a core (cores 4–7).
void matrix_multiplication(const float* A, const float* B, float* C) {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // Pin OpenMP thread to core 4, 5, 6, or 7
        int core_id = 4 + thread_id;  // thread_id from 0 to 3 gives core 4 to 7
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t current_thread = pthread_self();
        int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
        if (rc != 0) {
            std::cerr << "Error setting affinity for OpenMP thread " << thread_id << ": " << rc << "\n";
        }
        
        // Use OpenMP for loop to divide work among threads.
        #pragma omp for schedule(static)
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                float sum = 0.0f;
                for (int k = 0; k < SIZE; k++) {
                    sum += A[i * SIZE + k] * B[k * SIZE + j];
                }
                C[i * SIZE + j] = sum;
            }
        }
    }
}

int main() {
    // Allocate memory for matrices A, B, and C.
    float* A = new float[SIZE * SIZE];
    float* B = new float[SIZE * SIZE];
    float* C = new float[SIZE * SIZE];
    
    // Initialize matrices A and B with 1.0 (you can change these values as needed).
    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }
    
    // Set OpenMP to use 4 threads.
    omp_set_num_threads(4);
    
    // Create 7 send threads that simulate the send() function on cores 0–3.
    std::vector<std::thread> send_threads;
    for (int i = 0; i < 7; i++) {
        send_threads.emplace_back(send_thread_func, i);
    }
    
    // Start timing the matrix multiplication.
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Perform the matrix multiplication (on cores 4–7).
    matrix_multiplication(A, B, C);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    
    // Stop the send threads.
    running.store(false);
    for (auto& th : send_threads) {
        th.join();
    }
    
    std::cout << "Matrix multiplication took " << duration.count() << " seconds." << std::endl;
    
    // Free allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}
