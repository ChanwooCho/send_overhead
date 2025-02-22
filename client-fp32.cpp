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
#include <sched.h>        // For setting CPU affinity
#include <sys/syscall.h>  // For getting thread id
#include <errno.h>
#include <string>

// Matrix dimensions: we multiply a 5120x5120 matrix with a 5120x1 vector.
#define ROWS 5120
#define COLS 5120

// Size of the message to send (1KB).
#define ONE_KB 1024

// Structure to pass parameters to the async send thread.
struct AsyncSendParams {
    int sockfd;         // TCP socket
    int core_id;        // Which core to use for sending
    char* message;      // Message to send
    size_t msg_len;     // Length of the message
};

// This function runs in a separate thread and sends a 1KB message.
void* async_send(void* arg) {
    AsyncSendParams* params = (AsyncSendParams*) arg;
    
    // Set the CPU for this thread.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(params->core_id, &cpuset);
    pid_t tid = syscall(SYS_gettid);
    if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
        // Error setting affinity (commented out for simplicity)
    }
    
    // Send the message (blocking call).
    ssize_t bytes_sent = send(params->sockfd, params->message, params->msg_len, 0);
    
    // Free the message and parameters.
    free(params->message);
    delete params;
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    // Usage: client <ip_address:port>
    if (argc != 2) {
        std::cerr << "Usage: client <ip_address:port>" << std::endl;
        return -1;
    }
    
    // Parse the IP and port.
    std::string input(argv[1]);
    std::size_t colon_pos = input.find(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Invalid argument format. Use: <ip_address:port>" << std::endl;
        return -1;
    }
    std::string server_ip = input.substr(0, colon_pos);
    int server_port = std::stoi(input.substr(colon_pos + 1));
    std::cout << "Server IP: " << server_ip << ", Port: " << server_port << std::endl;
    
    // Print available cores.
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout << "Number of available cores: " << num_cores << std::endl;
    
    // Allocate memory:
    // A is a 5120x5120 matrix, B is a 5120x1 vector, C is the result vector.
    float* A = new float[ROWS * COLS];
    float* B = new float[COLS];
    float* C = new float[ROWS];

    // Initialize A and B with random float values.
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i * COLS + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    for (int i = 0; i < COLS; i++) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // Use 4 OpenMP threads.
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        // Map each multiplication thread to cores 4, 5, 6, 7.
        int desired_core = thread_id + 4;
        if (desired_core < num_cores) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(desired_core, &cpuset);
            pid_t tid = syscall(SYS_gettid);
            if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
                std::cerr << "Error setting thread affinity for thread " 
                          << thread_id << ": " << strerror(errno) << std::endl;
            }
        }
        
        // Create a TCP socket and connect to the server.
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
        
        // Divide the work between threads.
        int duty = ROWS / 4;
        int start = thread_id * duty;
        int end = (thread_id + 1) * duty;
        pthread_t send_thread;
        bool async_send_started = false;
        double start_time = omp_get_wtime();
        
        // For each row in the assigned range, compute the dot product.
        for (int i = start; i < end; i++) {
            // At the middle row for thread 3, start the async TCP send.
            if (!async_send_started && i == start + (duty / 2) && thread_id == 3) {
                async_send_started = true;
                char* message = (char*)malloc(ONE_KB);
                memset(message, 'A', ONE_KB);
                
                AsyncSendParams* send_params = new AsyncSendParams;
                send_params->sockfd = sockfd;
                send_params->core_id = thread_id; // Use cores 0-3 for async send.
                send_params->message = message;
                send_params->msg_len = ONE_KB;
                
                int rc = pthread_create(&send_thread, nullptr, async_send, (void*) send_params);
                if (rc != 0) {
                    std::cerr << "Thread " << thread_id << " failed to create async send thread: " 
                              << strerror(rc) << std::endl;
                } else {
                    std::cout << "Thread " << thread_id 
                              << " started async send thread at row " << i << std::endl;
                }
            }
            
            float sum = 0.0f;
            // Use SIMD to help the compiler vectorize this loop.
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < COLS; k++) {
                sum += A[i * COLS + k] * B[k];
            }
            C[i] = sum;
        }
        
        // Wait for the asynchronous send thread to finish.
        if (async_send_started) {
            pthread_join(send_thread, nullptr);
            std::cout << "Thread " << thread_id << " async send thread completed." << std::endl;
        }
        
        double thread_time = omp_get_wtime() - start_time;
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " execution time: " 
                      << thread_time * 1000 * 1000 << " us" << std::endl;
        }
        
        close(sockfd);
    }
    
    // Print the first 10 results of the result vector.
    std::cout << "First 10 results:" << std::endl;
    for (int i = 0; i < 10 && i < ROWS; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;
    
    // Free allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}
