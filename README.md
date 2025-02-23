g++ -fopenmp -O2 client-int8.cpp -o client
g++ -fopenmp -O2 client-fp32.cpp -o client
g++ server.cpp -o server

./server 9998
./client-int8 0 23 192.168.xxx.xxx:9998

1st config  0 -> only matmul
            1 -> send() in the middle of the matmul

2nd config 23 -> # of heads

3rd config 9998 -> port

해당 코드는 (128 x # of heads) X 5120 matmul 5120 X 1 의 행렬 연산에 관한 것이다.
행렬 연산은 Tiled Matrix Multiplication을 사용 (tile size는 5 x 1)
Matmul을 진행할 때는 4-7 cores를 사용하고
send()를 진행할 때는 1-3 cores를 사용한다 (즉 matmul 도중에 3번 send() matmul 끝나고 1번 send() 진행하는 scenario)

