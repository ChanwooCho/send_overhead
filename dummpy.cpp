#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sched.h>
#include <omp.h>

#define SIZE 4096  // Matrix size (4096 x 4096)

// Dummy send() function to simulate the send overhead.
// Replace this with your actual send() implementation if needed.
void send_data() {
    volatile int dummy = 0;
    // A simple loop to simulate some work
    for (int i = 0; i < 100000; i++) {
        dummy += i;
    }
}

// Structure for thread argument to pass how many times to call send_data()
typedef struct {
    int times;
} thread_arg;

// Thread function for asynchronous send() calls.
void* send_thread_func(void* arg) {
    // Pin this thread to core 0 (or any core not used for matrix multiplication)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    pthread_t thread = pthread_self();
    pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);

    thread_arg *targ = (thread_arg*)arg;
    for (int i = 0; i < targ->times; i++) {
        send_data();
    }
    return NULL;
}

int main() {
    // Allocate memory for matrices A, B, and C.
    float *A = (float*) malloc(SIZE * SIZE * sizeof(float));
    float *B = (float*) malloc(SIZE * SIZE * sizeof(float));
    float *C = (float*) malloc(SIZE * SIZE * sizeof(float));
    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices: A and B with 1.0, C with 0.0.
    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
        C[i] = 0.0f;
    }

    // Create an asynchronous thread for send() calls.
    pthread_t send_thread;
    thread_arg targ;
    targ.times = 7;  // Call send_data() 7 times

    if (pthread_create(&send_thread, NULL, send_thread_func, &targ) != 0) {
        printf("Failed to create send thread\n");
        return -1;
    }

    // Disable dynamic adjustment of threads (optional).
    omp_set_dynamic(0);
    // Set OpenMP to use 4 threads.
    omp_set_num_threads(4);

    // Start timing the matrix multiplication.
    double start_time = omp_get_wtime();

    // Matrix multiplication using OpenMP.
    // Each thread is pinned to one of the cores 4, 5, 6, or 7.
    #pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        int core_id = 4 + thread_num;  // Use cores 4,5,6,7 for computation.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pthread_t tid = pthread_self();
        pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);

        // Use a collapsed loop for better load balance.
        #pragma omp for collapse(2) schedule(static)
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

    // End timing after matrix multiplication completes.
    double end_time = omp_get_wtime();
    printf("Matrix multiplication took %f seconds\n", end_time - start_time);

    // Wait for the asynchronous send thread to finish.
    pthread_join(send_thread, NULL);

    // Optional: Verify a result.
    // Since A and B are filled with ones, each element in C should equal SIZE.
    if (C[0] != (float)SIZE) {
        printf("Unexpected result: C[0] = %f\n", C[0]);
    } else {
        printf("Result verified: C[0] = %f\n", C[0]);
    }

    // Free allocated memory.
    free(A);
    free(B);
    free(C);

    return 0;
}
