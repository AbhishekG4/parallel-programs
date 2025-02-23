// Engineer: Abhishek Gautam
// Last Updated: 02-22-2025

// Problem/Functionality: This is a Kernel to perform a reduction on an input
//                        array. In this case we are simply going to sum all
//                        elements of the array.

// High-level Approach:
// 1) The array is to be divided into segments processed by thread blocks of
//    a particular maximum size.
// 2) A tree based approach is used within each thread block.
//     2.1) Each block has as many elements to work with as threads.
//     2.2) A loop is run in which in each cycle the threads reduce the
//          block data to half its size until only 1 element is left which
//          is the result of the reduction on the block
//     2.3) The block's result is stored at its block index in the input
//          array.
// 3) The problem size has now shrunk from N to no. of blocks launced. This
//    process along with the kernel launch is repeated in a loop on the
//    results of the thread blocks until 1 thread block is left which
//    eveluates to one result which is the final result.
// 4) As an optimization, each thread brings in two values from the input
//    array into shared memory. Thus, each block processes twice the size
//    of input elements. This helps distribute work among the threads more
//    evenly threads start to become idle as the process continues.
// 5) As an optimization, the last warp(32 threads) worth of work is
//    unrolled. This helps avoid instruction overhead and reduces useless
//    work in the other warps.

// =============================================================================

#include <stdio.h>
#include <stdlib.h>

#define N_ (64 * 1024 * 1024)
#define MAX_THREADS 512

typedef double dtype;

int nxtPow2(int x) {  // Obtains the next greatest power of 2 if not already a
                      // power of 2
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}
void GetNumBlocksAndThreads(unsigned int n, int& blocks,
                            int& threads) {  // Assuming first add. Each thread
                                             // brings in two values from mem
  threads = (n > MAX_THREADS * 2) ? MAX_THREADS : nxtPow2((n + 1) / 2);
  blocks = (n + threads * 2 - 1) / (threads * 2);
}

__device__ void WarpUnroll(volatile dtype* blk_mem, unsigned int tid) {
  blk_mem[tid] += blk_mem[tid + 32];
  blk_mem[tid] += blk_mem[tid + 16];
  blk_mem[tid] += blk_mem[tid + 8];
  blk_mem[tid] += blk_mem[tid + 4];
  blk_mem[tid] += blk_mem[tid + 2];
  blk_mem[tid] += blk_mem[tid + 1];
}
__global__ void ReductionKernel(dtype* i_data, unsigned int n, unsigned int N) {
  __shared__ dtype blk_mem[MAX_THREADS];
  unsigned int bid = blockIdx.y * gridDim.x + blockIdx.x;
  unsigned int idx = bid * 2 * blockDim.x + threadIdx.x;

  // bringing in main memory to block memory
  if (idx >= N)
    blk_mem[threadIdx.x] = 0.0;
  else if (idx + blockDim.x >= N)
    blk_mem[threadIdx.x] = i_data[idx];
  else
    blk_mem[threadIdx.x] =
        i_data[idx] + i_data[idx + blockDim.x];  // Each thread brings in two
                                                 // elemnts by adding them first
  __syncthreads();

  // Reducing block
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (threadIdx.x < s) blk_mem[threadIdx.x] += blk_mem[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x < 32) WarpUnroll(blk_mem, threadIdx.x);

  // writing back to main memory
  if (threadIdx.x == 0) i_data[bid] = blk_mem[0];
}

dtype Reduction(dtype* i_data_h, unsigned int N) {
  dtype* i_data_d;
  // Move data to GPU
  cudaMalloc(&i_data_d, sizeof(dtype) * N);
  cudaMemcpy(i_data_d, i_data_h, sizeof(dtype) * N, cudaMemcpyHostToDevice);
  // Reduce
  unsigned int n = N;
  int threads = 0;
  int blocks = 0;
  while (n > 1) {
    GetNumBlocksAndThreads(n, blocks, threads);
    // Kernel launch
    dim3 grid_block(16, (blocks + 16 - 1) / 16);
    dim3 thread_block(threads, 1);
    ReductionKernel<<<grid_block, thread_block>>>(i_data_d, n, N);
    n = blocks;  // new problem size
  }
  // Move result back
  cudaMemcpy(i_data_h, i_data_d, sizeof(dtype) * 1, cudaMemcpyDeviceToHost);
  // Free
  cudaFree(i_data_d);

  return i_data_h[0];
}

dtype CPUReduction(dtype* i_data, unsigned int N) {
  dtype sum = 0;
  for (int i = 0; i < N; i++) {
    sum += i_data[i];
  }
  return sum;
}

int main(int argc, char** argv) {
  unsigned int N;
  dtype* i_data_h;

  // Procure arguments from user
  if (argc > 1) {
    N = (unsigned int)atoi(argv[1]);
    printf("N = %u\n", N);
  } else {
    N = N_;
    printf("N = %u\n", N);
  }

  // Prepare random input
  i_data_h = (dtype*)malloc(sizeof(dtype) * N);
  if (!i_data_h) printf("--Malloc failed for i_data_h--\n");
  srand48(21);
  for (int i = 0; i < N; i++) i_data_h[i] = drand48() / 100000;

  // Reduce
  dtype ground_truth = CPUReduction(i_data_h, N);
  dtype device_result = Reduction(i_data_h, N);
  // Verify
  printf("device result = %f\nground  truth = %f\n", device_result,
         ground_truth);
  // Free
  free(i_data_h);
}