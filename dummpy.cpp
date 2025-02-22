#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define ROWS 5120  // 행렬 A의 행 수
#define COLS 5120  // 행렬 A의 열 수, 행렬 B의 행 수

int main() {
    // 행렬 A (5120 x 5120)를 동적 할당
    double** A = new double*[ROWS];
    for (int i = 0; i < ROWS; ++i) {
        A[i] = new double[COLS];
    }
    
    // 행렬 B (5120 x 1)를 동적 할당
    double* B = new double[COLS];
    
    // 결과 행렬 C (5120 x 1)를 동적 할당
    double* C = new double[ROWS];
    
    // 난수 초기화
    srand(time(NULL));
    
    // 행렬 A 초기화 (예: 0~9 사이의 난수)
    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            A[i][j] = rand() % 10;
        }
    }
    
    // 행렬 B 초기화 (예: 0~9 사이의 난수)
    for (int i = 0; i < COLS; ++i) {
        B[i] = rand() % 10;
    }
    
    // OpenMP를 사용해 4개의 스레드로 설정
    omp_set_num_threads(4);
    
    // 행렬 곱셈: C = A * B
    // 각 C[i]는 A[i]와 B의 내적입니다.
    #pragma omp parallel for
    printf("%d\n", omp_get_thread_num());
    for (int i = 0; i < ROWS; ++i) {
        double sum = 0.0;
        for (int j = 0; j < COLS; ++j) {
            sum += A[i][j] * B[j];
        }
        C[i] = sum;
    }
    
    // 결과 확인: 처음 10개 요소 출력
    for (int i = 0; i < 10; ++i) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }
    
    // 동적 할당한 메모리 해제
    for (int i = 0; i < ROWS; ++i) {
        delete [] A[i];
    }
    delete [] A;
    delete [] B;
    delete [] C;
    
    return 0;
}
