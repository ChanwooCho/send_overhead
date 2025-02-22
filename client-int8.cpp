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

// Matrix dimensions.
#define ROWS 5120
#define COLS 5120
#define B_COLS 1

// Define the size of the message to send (1KB).
#define ONE_KB 1024

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
        // std::cerr << "Error setting affinity in async_send: " 
        //           << strerror(errno) << std::endl;
    } else {
        // std::cout << "Async send thread is set to core " << params->core_id << std::endl;
    }
    
    // Perform the blocking send of 1KB data.
    ssize_t bytes_sent = send(params->sockfd, params->message, params->msg_len, 0);
    // if (bytes_sent < 0) {
    //     std::cerr << "Send error in async thread: " << strerror(errno) << std::endl;
    // } else {
    //     std::cout << "Async send thread on core " << params->core_id 
    //               << " sent " << bytes_sent << " bytes." << std::endl;
    // }
    
    // Free allocated resources.
    free(params->message);
    delete params;
    pthread_exit(nullptr);
}

int main(int argc, char* argv[]) {
    // Usage: client <send -> true(1) false(0)> <ip_address:port>
    if (argc != 3) {
        std::cerr << "Usage: client <ip_address:port>" << std::endl;
        return -1;
    }
    
    // Parse the IP address and port.
    std::string input(argv[2]);
    std::size_t colon_pos = input.find(':');
    if (colon_pos == std::string::npos) {
        std::cerr << "Invalid argument format. Use: <ip_address:port>" << std::endl;
        return -1;
    }
    std::string server_ip = input.substr(0, colon_pos);
    int server_port = std::stoi(input.substr(colon_pos + 1));
    int send_overhead = std::atoi(argv[1]);
    std::cout << "Server IP: " << server_ip << ", Port: " << server_port << std::endl;
    
    // Print the number of available cores.
    int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    std::cout << "Number of available cores: " << num_cores << std::endl;
    
    // Allocate memory for matrices A, B, and C.
    // A and B are int8 matrices; C is int32 to store the accumulated result.
    int8_t* A = new int8_t[ROWS * COLS];
    int8_t* B = new int8_t[COLS * B_COLS];
    int32_t* C = new int32_t[ROWS * B_COLS];

    // Initialize matrices A and B with random int8_t values.
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

    // Start the OpenMP parallel region.
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        // Map matrix multiplication threads to cores 4, 5, 6, and 7.
        int desired_core = thread_id + 4;
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
        
        // Each thread creates its own TCP socket and connects to the server.
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
        
        // Determine the range of rows this thread will process.
        int duty = ROWS / 4;
        int start = thread_id * duty;
        int end = (thread_id + 1) * duty;
        
        pthread_t send_thread;
        bool async_send_started = false;
        int actual_cpu = sched_getcpu();
        double start_time = omp_get_wtime();
        
        // Matrix multiplication loop.
        for (int i = start; i < end; i++) {
            // At the halfway point, launch an asynchronous TCP send of 1KB.
            if (!async_send_started && i == start + (duty / 4 * thread_id) && send_overhead) {
                async_send_started = true;
                // Create a 1KB message filled with 'A'.
                char* message = (char*)malloc(ONE_KB / 4);
                memset(message, 'A', ONE_KB / 4);
                
                // Allocate and set parameters for the async send thread.
                AsyncSendParams* send_params = new AsyncSendParams;
                send_params->sockfd = sockfd;
                send_params->core_id = thread_id; // Use cores 0-3 for async send.
                send_params->message = message;
                send_params->msg_len = ONE_KB / 4;
                
                int rc = pthread_create(&send_thread, nullptr, async_send, (void*) send_params);
                // if (rc != 0) {
                //     std::cerr << "Thread " << thread_id << " failed to create async send thread: " 
                //               << strerror(rc) << std::endl;
                // } else {
                //     std::cout << "Thread " << thread_id 
                //               << " started async send thread at row " << i << std::endl;
                // }
            }
            // Standard matrix multiplication inner loop.
            for (int j = 0; j < B_COLS; j++) {
                int32_t sum = 0;
                for (int k = 0; k < COLS; k++) {
                    // Multiply int8_t values and accumulate into an int32_t sum.
                    sum += static_cast<int32_t>(A[i * COLS + k]) * static_cast<int32_t>(B[k * B_COLS + j]);
                }
                C[i * B_COLS + j] = sum;
            }
        }
        
        // Wait for the asynchronous send thread to finish, if it was started.
        if (async_send_started) {
            pthread_join(send_thread, nullptr);
            // std::cout << "Thread " << thread_id << " async send thread completed." << std::endl;
        }
        
        double thread_time = omp_get_wtime() - start_time;
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " execution time: " 
                      << thread_time * 1000 * 1000 << " us" << std::endl;
            std::cout << "Thread " << thread_id 
                      << " is actually running on CPU " << actual_cpu << std::endl;
        }
        
        close(sockfd);
    }
    
    // Print the first 10 results of matrix C.
    std::cout << "First 10 results of matrix C:" << std::endl;
    for (int i = 0; i < 10 && i < ROWS * B_COLS; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;
    
    // Clean up allocated memory.
    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}
