#include <iostream>
#include <random>
#include <vector>
#include <boost/timer/timer.hpp>

__global__ void kernel(int * a, int * b, int count) {
  int offset = threadIdx.x + blockDim.x * blockIdx.x;
  for (int index = offset; index < count; index += blockDim.x * gridDim.x) {
    b[index] = 2 * a[index];
  }
}

std::vector<int> ver1(const std::vector<int> &vec, const int N) {
  int *a_d = nullptr;
  cudaMalloc((void**)&a_d, sizeof(int) * N);
  int *b_d = nullptr;
  cudaMalloc((void**)&b_d, sizeof(int) * N);
  cudaMemcpy(a_d, vec.data(), sizeof(int) * N, cudaMemcpyHostToDevice);
  kernel<<<1024, 256>>>(a_d, b_d, N);
  std::vector<int> b(N);
  cudaMemcpy(b.data(), b_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(b_d);
  return b;
}

std::vector<int> ver2(const std::vector<int> &vec, const int N, const int n) {
  int *a_h = nullptr;
  cudaMallocHost((void**)&a_h, sizeof(int) * N);
  int *b_h = nullptr;
  cudaMallocHost((void**)&b_h, sizeof(int) * N);
  int *a_d = nullptr;
  cudaMalloc((void**)&a_d, sizeof(int) * N);
  int *b_d = nullptr;
  cudaMalloc((void**)&b_d, sizeof(int) * N);
  memcpy(a_h, vec.data(), sizeof(int) * N);
  cudaStream_t str[3];
  for (int i = 0; i < 3; ++i) {
    cudaStreamCreate(str + i);
  }
  for (int i = 0; i < 3; ++i) {
    cudaMemcpyAsync(a_d + n*i, a_h + n*i, sizeof(int) * n, cudaMemcpyHostToDevice, str[i]);
  }
  for (int i = 0; i < 3; ++i) {
    kernel<<<1024, 256, 0, str[i]>>>(a_d + n*i, b_d + n*i, n);
  }
  for (int i = 0; i < 3; ++i) {
    cudaMemcpyAsync(b_h + n*i, b_d + n*i, sizeof(int) * n, cudaMemcpyDeviceToHost, str[i]);
  }
  for (int i = 0; i < 3; ++i) {
    cudaStreamSynchronize(str[i]);
    cudaStreamDestroy(str[i]);
  }
  std::vector<int> b(N);
  memcpy(b.data(), b_h, sizeof(int) * N);
  cudaFree(a_h);
  cudaFree(b_h);
  cudaFree(a_d);
  cudaFree(b_d);
  return b;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " N K" << std::endl;
    return -1;
  }
  int n = atoi(argv[1]);
  int k = atoi(argv[2]);
  int N = 3 * n;
  std::vector<int> vec(N);
  std::random_device rd;
  std::mt19937 mt(rd());
  for (int i = 0; i < N; ++i) {
    vec[i] = mt();
  }
  boost::timer::cpu_timer timer;
  std::vector<int> b;
  switch (k) {
    case 0: b = ver1(vec, N); break;
    case 1: b = ver2(vec, N, n); break;
  }
  std::cout << timer.format() << std::endl;
  int diff = 0;
  for (int i = 0; i < N; ++i) {
    diff += abs(vec[i] * 2 - b[i]);
  }
  std::cerr << "diff: " << diff << std::endl;
  return 0;
}
