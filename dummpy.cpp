#include <iostream>
#include <omp.h>
#include <chrono>
#include <thread>
#include <vector>
#include <atomic>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <errno.h>
#include <cstring>

#define SIZE 4096  // Matrix size: 4096 x 4096

#if defined(__ANDROID__)
    #if defined(__x86_64__)
    #define __NR_sched_setaffinity 203
    #elif defined(__arm__)
    #define __NR_sched_setaffinity 241
    #elif defined(__aarch64__)
    #define __NR_sched_setaffinity 122
    #endif
    #define CPU_SETSIZE 1024
    #define __NCPUBITS (8 * sizeof (unsigned long))
    typedef struct {
        unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
    } cpu_set_t;

    void CPU_ZERO(cpu_set_t *set) {
        memset(set, 0, sizeof(cpu_set_t));
    }

    void CPU_SET(int cpu, cpu_set_t *set) {
        set->__bits[cpu / __NCPUBITS] |= (1UL << (cpu % __NCPUBITS));
    }

    // Define sched_setaffinity using syscall
    int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask) {
        int result = syscall(__NR_sched_setaffinity, pid, cpusetsize, mask);
        if (result != 0) {
            errno = result;
            return -1;
        }
        return 0;
    }
#endif

// A flag to signal the send threads to stop.
std::atomic<bool> running(true);

// This function simulates a "send" operation.
// It does some dummy work to create an overhead.
void send_function(int id) {
    volatile int dummy = 0;
    // Loop to simulate work (adjust iterations as needed)
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

    // Get the thread id using syscall and set its affinity.
    pid_t tid = syscall(SYS_gettid);
    if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Warning: Error setting affinity for send thread " 
                  << thread_id << ": " << strerror(errno) << "\n";
    }
    
    // Keep calling send_function while matrix multiplication is running.
    while (running.load()) {
        send_function(thread_id);
    }
}

// This function performs matrix multiplication of two SIZE x SIZE matrices.
// It uses an OpenMP parallel region where each thread is intended to be pinned to a core (cores 4–7).
void matrix_multiplication(const float* A, const float* B, float* C) {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        // Pin OpenMP thread to core 4, 5, 6, or 7
        int core_id = 4 + thread_id;  // thread_id from 0 to 3 gives core 4 to 7
        cpu_set_t set;
        
        printf("tid = %d, cid = %d\n", thread_id, core_id);
        
        CPU_ZERO(&set);
        CPU_SET(core_id, &set);
        // Get the thread id and set its affinity.
        pid_t tid = syscall(SYS_gettid);
        if (sched_setaffinity(0, sizeof(cpu_set_t), &set) != 0) {
            std::cerr << "Warning: Error setting affinity for OpenMP thread " 
                      << thread_id << ": " << strerror(errno) << "\n";
        }
        
        // Divide work among threads.
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
    
    // Initialize matrices A and B with 1.0.
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
    
    // Perform the matrix multiplication (intended to run on cores 4–7).
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
