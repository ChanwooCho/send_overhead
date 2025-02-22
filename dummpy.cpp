#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>

#define SIZE 4096

// Dummy send() function to simulate overhead.
// Replace this with your actual send() call if needed.
void send_function() {
    volatile int sum = 0;
    for (int i = 0; i < 100000; i++) {
        sum += i;
    }
}

// Set thread affinity to a specific core.
void set_affinity(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_t thread = pthread_self();
    int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        fprintf(stderr, "Error setting affinity on core %d: %s\n", core_id, strerror(ret));
    }
}

// Function executed by each send() thread.
void* send_thread_func(void* arg) {
    // Pin this thread to core 0 (or any other core not used by matrix multiplication)
    set_affinity(0);
    send_function();
    return NULL;
}

int main() {
    // Allocate matrices as 1D arrays (we use row-major order)
    float *A = (float*) malloc(SIZE * SIZE * sizeof(float));
    float *B = (float*) malloc(SIZE * SIZE * sizeof(float));
    float *C = (float*) malloc(SIZE * SIZE * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices A and B with random float values.
    for (int i = 0; i < SIZE * SIZE; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    // Create 7 threads that call send_function asynchronously.
    pthread_t send_threads[7];
    for (int i = 0; i < 7; i++) {
        if (pthread_create(&send_threads[i], NULL, send_thread_func, NULL)) {
            fprintf(stderr, "Failed to create send thread %d\n", i);
            return EXIT_FAILURE;
        }
    }

    // Set OpenMP to use 4 threads for matrix multiplication.
    omp_set_num_threads(4);

    // Start timer.
    struct timeval start, end;
    gettimeofday(&start, NULL);

    // Matrix multiplication using OpenMP.
    // Each thread will pin itself to one of the cores 4,5,6,7.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        set_affinity(4 + tid);  // Thread 0->core4, 1->core5, etc.

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

    // End timer.
    gettimeofday(&end, NULL);
    double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 + 
                        (end.tv_usec - start.tv_usec) / 1000.0;
    printf("Matrix multiplication took: %f ms\n", elapsed_ms);

    // Wait for send() threads to complete.
    for (int i = 0; i < 7; i++) {
        pthread_join(send_threads[i], NULL);
    }

    // Free allocated memory.
    free(A);
    free(B);
    free(C);

    return 0;
}
