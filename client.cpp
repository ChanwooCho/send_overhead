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

// Matrix dimensions.
#define ROWS 5120
#define COLS 5120
#define B_COLS 1

// Define the size of the message to send (1KB).
#define ONE_KB 1024

// Data structure for the pre-created asynchronous send worker.
struct AsyncWorkerData {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool request;   // True when a send request is waiting.
    bool exit;      // True when the worker should exit.
    int sockfd;     // Socket descriptor.
    int core_id;    // Core to set affinity.
    char* message;  // Pointer to the 1KB message.
    size_t msg_len; // Length of the message.
};

// The worker thread function. It waits until a request is signaled.
void* async_worker(void* arg) {
    AsyncWorkerData* data = (AsyncWorkerData*) arg;
    while (true) {
        // Wait for a send request or exit signal.
        pthread_mutex_lock(&data->mutex);
        while (!data->request && !data->exit) {
            pthread_cond_wait(&data->cond, &data->mutex);
        }
        // If exit flag is set, break out.
        if (data->exit) {
            pthread_mutex_unlock(&data->mutex);
            break;
        }
        // Copy local variables for sending.
        int sockfd = data->sockfd;
        int core_id = data->core_id;
        char* message = data->message;
        size_t msg_len = data->msg_len;
        // Clear the request flag.
        data->request = false;
        pthread_mutex_unlock(&data->mutex);
        
        // Set the CPU affinity for this worker thread.
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        pid_t tid = syscall(SYS_gettid);
        if (sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset) != 0) {
            std::cerr << "Error setting affinity in async_worker: " 
                      << strerror(errno) << std::endl;
        } else {
            std::cout << "Async worker thread is set to core " << core_id << std::endl;
        }
        
        // Perform the blocking send of 1KB data.
        ssize_t bytes_sent = send(sockfd, message, msg_len, 0);
        if (bytes_sent < 0) {
            std::cerr << "Send error in async_worker: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Async worker thread on core " << core_id 
                      << " sent " << bytes_sent << " bytes." << std::endl;
        }
        
        // Free the allocated message buffer.
        free(message);
    }
    return nullptr;
}

int main(int argc, char* argv[]) {
    // Usage: client <ip_address:port>
    if (argc != 2) {
        std::cerr << "Usage: client <ip_address:port>" << std::endl;
        return -1;
    }
    
    // Parse the IP address and port.
    std::string input(argv[1]);
    size_t colon_pos = input.find(':');
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
    float* A = new float[ROWS * COLS];
    float* B = new float[COLS * B_COLS];
    float* C = new float[ROWS * B_COLS];

    // Initialize matrices A and B with random values.
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

    // Create a 1KB message filled with 'A'.
    char* message = (char*)malloc(ONE_KB);
    memset(message, 'A', ONE_KB);
    
    // Fill in the worker data and signal the async worker.
    pthread_mutex_lock(&worker_data.mutex);
    worker_data.sockfd = sockfd;
    worker_data.core_id = thread_id; // Using thread id for affinity.
    worker_data.message = message;
    worker_data.msg_len = ONE_KB;
    worker_data.request = true;
    
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
            } else {
                std::cout << "Multiplication thread " << thread_id 
                          << " is set to core " << desired_core << std::endl;
            }
        } else {
            std::cerr << "Warning: Not enough cores for thread " 
                      << thread_id << " to be set on core " << desired_core << std::endl;
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
        if (connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cerr << "Thread " << thread_id << " connection failed: " 
                      << strerror(errno) << std::endl;
            close(sockfd);
            #pragma omp cancel parallel
        }
        std::cout << "Thread " << thread_id << " connected to server." << std::endl;
        
        // For thread 3, pre-create the asynchronous send worker thread.
        bool async_send_started = false;
        pthread_t worker_thread;
        AsyncWorkerData worker_data;
        if (thread_id == 3) {
            worker_data.request = false;
            worker_data.exit = false;
            pthread_mutex_init(&worker_data.mutex, nullptr);
            pthread_cond_init(&worker_data.cond, nullptr);
            int rc = pthread_create(&worker_thread, nullptr, async_worker, (void*)&worker_data);
            if (rc != 0) {
                std::cerr << "Thread " << thread_id << " failed to create async worker thread: " 
                          << strerror(rc) << std::endl;
            }
        }
        
        // Determine the range of rows this thread will process.
        int duty = ROWS / 4;
        int start = thread_id * duty;
        int end = (thread_id + 1) * duty;
        std::cout << "Thread " << thread_id << ": processing rows " 
                  << start << " to " << end - 1 << std::endl;
        
        double start_time = omp_get_wtime();
        
        // Matrix multiplication loop.
        for (int i = start; i < end; i++) {
            // For thread 3, at the halfway point, signal the worker to send data.
            if (thread_id == 3 && !async_send_started && i == start + (duty / 2)) {
                async_send_started = true;
                
                pthread_cond_signal(&worker_data.cond);
                pthread_mutex_unlock(&worker_data.mutex);
                
                std::cout << "Thread " << thread_id 
                          << " signaled async worker thread at row " << i << std::endl;
            }
            // Standard matrix multiplication inner loop.
            for (int j = 0; j < B_COLS; j++) {
                float sum = 0.0f;
                for (int k = 0; k < COLS; k++) {
                    sum += A[i * COLS + k] * B[k * B_COLS + j];
                }
                C[i * B_COLS + j] = sum;
            }
        }
        
        double thread_time = omp_get_wtime() - start_time;
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " execution time: " 
                      << thread_time * 1000 * 1000 << " us" << std::endl;
        }
        
        // For thread 3, after finishing work, signal the worker to exit and join it.
        if (thread_id == 3 && async_send_started) {
            pthread_mutex_lock(&worker_data.mutex);
            worker_data.exit = true;
            pthread_cond_signal(&worker_data.cond);
            pthread_mutex_unlock(&worker_data.mutex);
            pthread_join(worker_thread, nullptr);
            std::cout << "Thread " << thread_id << " async worker thread completed." << std::endl;
            pthread_mutex_destroy(&worker_data.mutex);
            pthread_cond_destroy(&worker_data.cond);
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
