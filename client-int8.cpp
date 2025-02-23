#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>        // For sched_setaffinity and sched_getcpu
#include <sys/syscall.h>  // For SYS_gettid
#include <errno.h>
#include <string>
#include <cstdint>        // For int8_t and int32_t
#include <algorithm>      // For std::min

// Matrix dimensions.
#define ROWS 128
#define COLS 5120
#define B_COLS 1

// Define the size of the message to send (1KB).
#define ONE_KB 2560

// Structure to pass parameters to the asynchronous send thread.
struct AsyncSendParams {
    int sockfd;         // Socket descriptor for TCP connection.
    int core_id;        // Desired core (0-3) for async send.
    char* message;      // Message to send.
    size_t msg_len;     // Length of the message.
};

// Function that runs in a separate pthread to call send() asynchronously.
void* async_send(void* arg) {
    AsyncSendParams* params = (AsyncSendParams*) arg;
    
    // Set CPU affinity to the desired core (0-3).
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(params->core_id, &cpuset);
    pid_t tid = syscall(SYS_gettid);
    if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
        // Error handling can be added here if needed.
    }
    
    // Send 1KB data in a blocking call.
    ssize_t bytes_sent = send(params->sockfd, params->message, params->msg_len, 0);
    
    // Free the allocated memory.
    free(params->message);
    delete params;
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    // Usage: client <send_overhead (1 or 0)> <ip_address:port>
    if (argc != 4) {
        std::cerr << "Usage: client <send_overhead (1 or 0)> <# of heads> <ip_address:port>" << std::endl;
        return -1;
    }
    
    // Parse the IP address and port.
    int send_overhead = std::atoi(argv[1]);
    int num_head = std::atoi(argv[2]);
    std::string input(argv[3]);
    std::size_t colon_pos = input.find(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Invalid argument format. Use: <ip_address:port>" << std::endl;
        return -1;
    }
    std::string server_ip = input.substr(0, colon_pos);
    int server_port = std::stoi(input.substr(colon_pos + 1));

    std::cout << "Server IP: " << server_ip << ", Port: " << server_port << std::endl;
    
    // Print the number of available cores.
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout << "Number of available cores: " << num_cores << std::endl;
    
    // Allocate memory for matrices A, B, and C.
    int8_t* A = new int8_t[ROWS * num_head * COLS];
    int8_t* B = new int8_t[COLS * B_COLS];
    int32_t* C = new int32_t[ROWS * num_head * B_COLS];

    // Initialize matrices A and B with random int8_t values.
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < ROWS * num_head; i++) {
        for (int j = 0; j < COLS; j++) {
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
    
    // We will run the matrix multiplication 100 times.
    const int NUM_ITER = 100;
    const int NUM_THREADS = 4;
    // This array will hold each thread's execution time in one iteration.
    double thread_exec_time[NUM_THREADS] = {0};
    // This variable will sum the maximum time of each iteration.
    double global_time_sum = 0.0;
    
    // Start the OpenMP parallel region.
    #pragma omp parallel shared(global_time_sum, thread_exec_time, A, B, C, send_overhead, server_ip, server_port)
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();  // should be 4
        
        // (Optional) Set CPU affinity for multiplication threads.
        int desired_core = thread_id + 4; // Map threads to cores 4, 5, 6, 7.
        if (desired_core < num_cores) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(desired_core, &cpuset);
            pid_t tid = syscall(SYS_gettid);
            if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
                std::cerr << "Error setting multiplication thread affinity for thread " 
                          << thread_id << ": " << strerror(errno) << std::endl;
            }
        }
        
        // Create a TCP socket once per thread and connect to the server.
        int sockfd = socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            std::cerr << "Thread " << thread_id << " error creating socket: " 
                      << strerror(errno) << std::endl;
            #pragma omp cancel parallel
        }
        struct sockaddr_in serv_addr;
        memset(&serv_addr, 0, sizeof(serv_addr));
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(server_port);
        if (inet_pton(AF_INET, server_ip.c_str(), &serv_addr.sin_addr) <= 0) {
            std::cerr << "Thread " << thread_id << " invalid address: " << server_ip << std::endl;
            close(sockfd);
            #pragma omp cancel parallel
        }
        if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0 && thread_id == 3) {
            std::cerr << "Thread " << thread_id << " connection failed: " 
                      << strerror(errno) << std::endl;
            close(sockfd);
            #pragma omp cancel parallel
        }
        
        // Each thread works on a subset of rows.
        int duty = ROWS * num_head / num_threads;
        int start = thread_id * duty;
        int end = (thread_id + 1) * duty;
        
        // Repeat the matrix multiplication NUM_ITER times.
        for (int iter = 0; iter < NUM_ITER; iter++) {
            bool async_send_started = false;
            pthread_t send_thread;
            double start_time = omp_get_wtime();
            
            // Tiled matrix multiplication with a tile size of 5 x 1.
            const int TILE_ROWS = 5;
            const int TILE_COLS = 1; // Because B_COLS is 1.
            for (int ii = start; ii < end; ii += TILE_ROWS) {
                int i_max = std::min(ii + TILE_ROWS, end);
                for (int jj = 0; jj < B_COLS; jj += TILE_COLS) {
                    int j_max = std::min(jj + TILE_COLS, B_COLS);
                    for (int i = ii; i < i_max; i++) {
                        // Launch async send at a specific row.
                        if (!async_send_started && send_overhead && (thread_id != 3) && (i == start + (duty / 4 * (thread_id + 1)))) {
                            async_send_started = true;
                            printf("here!\n");
                            // Create a 1KB message filled with 'A'.
                            char* message = (char*)malloc(ONE_KB);
                            memset(message, 'A', ONE_KB);
                            
                            // Set parameters for the async send thread.
                            AsyncSendParams* send_params = new AsyncSendParams;
                            send_params->sockfd = sockfd;
                            send_params->core_id = thread_id; // Use cores 0-3 for async send.
                            send_params->message = message;
                            send_params->msg_len = ONE_KB;
                            
                            int rc = pthread_create(&send_thread, nullptr, async_send, (void*) send_params);
                            // You can check rc for errors if needed.
                        }
                        for (int j = jj; j < j_max; j++) {
                            int32_t sum = 0;
                            for (int k = 0; k < COLS; k++) {
                                sum += static_cast<int32_t>(A[i * COLS + k]) *
                                       static_cast<int32_t>(B[k * B_COLS + j]);
                            }
                            C[i * B_COLS + j] = sum;
                        }
                    }
                }
            }
            
            // Measure this thread's execution time.
            double thread_time = omp_get_wtime() - start_time;
            thread_exec_time[thread_id] = thread_time;

            // If an async send was started, wait for it to finish.
            if (async_send_started) {
                pthread_join(send_thread, nullptr);
            }
            
            // Wait for all threads.
            #pragma omp barrier
            
            // Only one thread (thread 0) finds the maximum time.
            #pragma omp single
            {
                double iter_max = thread_exec_time[0];
                for (int t = 1; t < num_threads; t++) {
                    if (thread_exec_time[t] > iter_max)
                        iter_max = thread_exec_time[t];
                }
                if (iter >= 10)
                    global_time_sum += iter_max;
                std::cout << "Iteration " << iter << " max time: " 
                          << iter_max * 1000000 << " us" << std::endl;
            }
            #pragma omp barrier
        }
        
        // Close the socket after all iterations.
        close(sockfd);
    } // End of parallel region.
    
    // Calculate and print the average matrix multiplication time.
    double avg_time = global_time_sum / (NUM_ITER - 10);
    std::cout << "Average matrix multiplication time over " << NUM_ITER 
              << " iterations: " << avg_time * 1000000 << " us" << std::endl;
    
    // Print the first 10 results of matrix C (from the last iteration).
    std::cout << "First 10 results of matrix C:" << std::endl;
    for (int i = 0; i < 10 && i < ROWS * num_head * B_COLS; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}
