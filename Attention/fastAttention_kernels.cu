#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <mma.h>
#include <sstream>
#include <stdio.h>
#include <torch/extension.h>
#include <vector>
using namespace nvcuda;

__global__ void matTransKernel(float* AT, float* A, int N);
__global__ void fusedAttnKernel(float* Q, float* K, float* V, float* O, int seq_len, int emb_dim, int tile_size);
__global__ void tcFusedAttnKernel(half* Q, half* K, half* V, half* O, int seq_len, int emb_dim, int tile_size);
__global__ void sparseTcFusedAttnKernel(half* Q, half* K, half* V, half* O, int seq_len, int emb_dim, int tile_size, int* row_ptr, int* col_idx);

void matTrans(torch::Tensor AT, torch::Tensor A)  {
  assert(AT.size(0) == AT.size(1));
  assert(AT.size(0) == A.size(0));
  assert(AT.size(1) == A.size(1));
  matTransKernel<<<1, 512>>>(AT.data_ptr<float>(), A.data_ptr<float>(), A.size(0));
}

void fusedAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int seq_len, int emb_dim, int tile_size){
  assert(Q.size(0) == K.size(0));
  //more asserts can be added
  dim3 gb(seq_len/tile_size,1);
  dim3 tb(32,4);
  int shared_mem_size = (4*tile_size*emb_dim + tile_size*tile_size + 64*tile_size + tile_size)*sizeof(float);
  fusedAttnKernel<<<gb,tb,shared_mem_size>>>(Q.data_ptr<float>(),K.data_ptr<float>(),V.data_ptr<float>(),O.data_ptr<float>(),seq_len,emb_dim,tile_size);
  cudaDeviceSynchronize();
}

void tcFusedAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int seq_len, int emb_dim, int tile_size){
  assert(Q.size(0) == K.size(0));
  //more asserts can be added
  dim3 gb(seq_len/tile_size,1);
  dim3 tb(32,4);
  int shared_mem_size = (4*tile_size*emb_dim + tile_size*tile_size)*sizeof(half) +
                        (64*tile_size + tile_size + tile_size*tile_size)*sizeof(float);
  tcFusedAttnKernel<<<gb, tb, shared_mem_size>>>(
    reinterpret_cast<__half*>(Q.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(K.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(V.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
    seq_len, emb_dim, tile_size
  );
  cudaDeviceSynchronize();
}

void sparseTcFusedAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int seq_len, int emb_dim, int tile_size, torch::Tensor row_ptr, torch::Tensor col_idx){
  assert(Q.size(0) == K.size(0));
  //more asserts can be added
  dim3 gb(seq_len/tile_size,1);
  dim3 tb(32,4);
  int shared_mem_size = (4*tile_size*emb_dim + tile_size*tile_size)*sizeof(half) +
                        (64*tile_size + tile_size + tile_size*tile_size)*sizeof(float) +
                        (seq_len+1)*sizeof(int);
  sparseTcFusedAttnKernel<<<gb, tb, shared_mem_size>>>(
    reinterpret_cast<__half*>(Q.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(K.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(V.data_ptr<at::Half>()),
    reinterpret_cast<__half*>(O.data_ptr<at::Half>()),
    seq_len, emb_dim, tile_size,
    row_ptr.data_ptr<int>(),
    col_idx.data_ptr<int>()
  );
  cudaDeviceSynchronize();
}


//=============================================Kernels============================================

__device__ void warpMatmul(float* A, float* B, float* C, int m, int n, int k, bool acc){
  for(unsigned int i = threadIdx.x; i<m*n; i += blockDim.x){
    unsigned int row = i/n;
    unsigned int col = i%n;

    float sum = 0.0;
    for(int j=0;j<k;j++) sum += A[row*k+j] * B[j*n+col];
    C[i] = (acc) ? C[i]+sum : sum;
  }
}

__device__ void warpReduce(float* A, volatile float* l, int seq_size){
  for(unsigned int i=threadIdx.x; i<seq_size; i += blockDim.x*2){
    float num2 = 0.0;
    if(i + blockDim.x < seq_size) num2 = A[i+blockDim.x];
    l[threadIdx.x] += A[i] + num2;
  }
  l[threadIdx.x] += l[threadIdx.x + 32];
  l[threadIdx.x] += l[threadIdx.x + 16];
  l[threadIdx.x] += l[threadIdx.x + 8];
  l[threadIdx.x] += l[threadIdx.x + 4];
  l[threadIdx.x] += l[threadIdx.x + 2];
  l[threadIdx.x] += l[threadIdx.x + 1];
}

__global__ void fusedAttnKernel(float* Q, float* K, float* V, float* O, int seq_len, int emb_dim, int tile_size){
  extern __shared__ float shm[];
  float* qShm = shm;
  float* kShm = qShm + tile_size*emb_dim;
  float* sShm = kShm + emb_dim*tile_size;
  float* vShm = sShm + tile_size*tile_size;
  float* oShm = vShm + tile_size*emb_dim;
  float* lShm = oShm + tile_size*emb_dim;
  float* l_prev = lShm + blockDim.x*2*tile_size;
  //l_prev shm size = tile_size

  unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
  unsigned int idx = blockDim.x*blockDim.y*blockIdx.x + tid;  //assumes linear block indexing
  unsigned int tbOffset = blockIdx.x*tile_size*emb_dim;
  int warpTileHeight = tile_size/blockDim.y;
  unsigned int warpQOffset = threadIdx.y*emb_dim*warpTileHeight;
  unsigned int warpSOffset = threadIdx.y*tile_size*warpTileHeight;

  //loading Q tile into shm
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) qShm[i] = Q[tbOffset+i];
  //initialize O
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) oShm[i] = 0.0;
  //initialize l_prev
  for(unsigned int i=tid; i<tile_size; i += blockDim.x*blockDim.y) l_prev[i] = 0.0;

  for(int iter=0; iter<seq_len/tile_size; iter++){
    unsigned int iterOffset = iter*tile_size*emb_dim;

    //loading K tile into shm transposed
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y){
      unsigned int row = i/emb_dim;
      unsigned int col = i%emb_dim;

      kShm[col*tile_size+row] = K[iterOffset + i];  //32-way banck conflict as of now
    }
    //loading V tile into shm
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) vShm[i] = V[iterOffset+i];
    __syncthreads();

    //warps multiply tiles
    warpMatmul(qShm+warpQOffset, kShm, sShm+warpSOffset, warpTileHeight, tile_size, emb_dim, false);
    __syncthreads();
    
    //exponentiate tile
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sShm[i] = __expf(sShm[i]);
    __syncthreads();

    //softmax reduction
    for(unsigned int i=tid; i<blockDim.x*2*tile_size; i += blockDim.x*blockDim.y) lShm[i] = 0.0;
    __syncthreads();
    for(unsigned int row = threadIdx.y; row<tile_size; row += blockDim.y) warpReduce(sShm+row*tile_size, lShm+row*blockDim.x*2, tile_size);
    __syncthreads();

    //applying corrected softmax
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y){
      int row = i/tile_size;
      sShm[i] = sShm[i]/(lShm[row*2*blockDim.x]+l_prev[row]); //first element of resp. row in l block (where elements have been reduced into)
    }
    //rescaling O
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y){
      int row = i/emb_dim;
      oShm[i] = (oShm[i]*l_prev[row])/(lShm[row*2*blockDim.x]+l_prev[row]);
    }
    //update l_prev
    for(unsigned int i = tid; i<tile_size; i += blockDim.x*blockDim.y) l_prev[i] += lShm[i*2*blockDim.x];
    __syncthreads();

    //warps multiply tiles of S and V
    warpMatmul(sShm+warpSOffset, vShm, oShm+warpQOffset, warpTileHeight, emb_dim, tile_size, true);
  }

  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) O[tbOffset + i] = oShm[i];
}

//====================== Tensor Core fused attn =========================
__device__ void tcMatmul(half* A, half* B, half* C, int m, int n, int k, int tile_size){
  int wid = threadIdx.y;
  // declare fragments
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

  int sub_blks_per_row = n/16;            //each warp is assigned a sub block
  int sub_blks_per_col = m/16;
  for(int i = wid; i<sub_blks_per_row*sub_blks_per_col; i += blockDim.y){
    int sub_blk_row = i/sub_blks_per_row;
    int sub_blk_col = i%sub_blks_per_row;
    wmma::load_matrix_sync(c_frag, C+sub_blk_row*16*n+sub_blk_col*16, n, wmma::mem_row_major);
    for(int j=0; j<k/16; j++){
      // load the inputs
      wmma::load_matrix_sync(a_frag, A+sub_blk_row*16*k+j*16, k);
      wmma::load_matrix_sync(b_frag, B+j*16*n+sub_blk_col*16, n);
      // perform the matrix multiplication
      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C+sub_blk_row*16*n+sub_blk_col*16, c_frag, n, wmma::mem_row_major);
  }
}

__global__ void tcFusedAttnKernel(half* Q, half* K, half* V, half* O, int seq_len, int emb_dim, int tile_size){
  extern __shared__ float float_shm[];  // declare shared memory as float first for alignment
  // allocate float-type shared memory first
  float* lShm = float_shm;
  float* l_prev = lShm + blockDim.x * 2 * tile_size;
  float* sfShm = l_prev + tile_size;
  // allocate half-type shared memory after the float section
  half* tc_shm = reinterpret_cast<half*>(sfShm + tile_size*tile_size);
  half* qShm = tc_shm;
  half* kShm = qShm + tile_size * emb_dim;
  half* sShm = kShm + emb_dim * tile_size;
  half* vShm = sShm + tile_size * tile_size;
  half* oShm = vShm + tile_size * emb_dim;
  //oShm size = tile_size*emb_dim

  unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
  unsigned int idx = blockDim.x*blockDim.y*blockIdx.x + tid;  //assumes linear block indexing
  unsigned int tbOffset = blockIdx.x*tile_size*emb_dim;
  int warpTileHeight = tile_size/blockDim.y;
  unsigned int warpQOffset = threadIdx.y*emb_dim*warpTileHeight;
  unsigned int warpSOffset = threadIdx.y*tile_size*warpTileHeight;

  //loading Q tile into shm
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) qShm[i] = Q[tbOffset+i];
  //initialize O
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) oShm[i] = __float2half(0.0);
  //initialize l_prev
  for(unsigned int i=tid; i<tile_size; i += blockDim.x*blockDim.y) l_prev[i] = 0.0;

  for(int iter=0; iter<seq_len/tile_size; iter++){
    unsigned int iterOffset = iter*tile_size*emb_dim;

    //loading K tile into shm transposed
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y){
      unsigned int row = i/emb_dim;
      unsigned int col = i%emb_dim;

      kShm[col*tile_size+row] = K[iterOffset + i];  //32-way banck conflict as of now
    }
    //loading V tile into shm
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) vShm[i] = V[iterOffset+i];
    //initializing S tile
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sShm[i] = __float2half(0.0);
    __syncthreads();

    //warps multiply tiles
    tcMatmul(qShm, kShm, sShm, tile_size, tile_size, emb_dim, tile_size);
    __syncthreads();

    //exponentiate tile
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sfShm[i] = __expf(__half2float(sShm[i]));
    //softmax reduction
    for(unsigned int i=tid; i<blockDim.x*2*tile_size; i += blockDim.x*blockDim.y) lShm[i] = 0.0;
    __syncthreads();
    for(unsigned int row = threadIdx.y; row<tile_size; row += blockDim.y) warpReduce(sfShm+row*tile_size, lShm+row*blockDim.x*2, tile_size);
    __syncthreads();
    //applying corrected softmax
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y){
      int row = i/tile_size;
      sfShm[i] = sfShm[i]/(lShm[row*2*blockDim.x]+l_prev[row]); //first element of resp. row in l block (where elements have been reduced into)
    }

    //rescaling O
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y){
      int row = i/emb_dim;
      oShm[i] = __float2half((__half2float(oShm[i])*l_prev[row])/(lShm[row*2*blockDim.x]+l_prev[row]));
    }
    //update l_prev
    for(unsigned int i = tid; i<tile_size; i += blockDim.x*blockDim.y) l_prev[i] += lShm[i*2*blockDim.x];
    //downscaling from float to half
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sShm[i] = __float2half(sfShm[i]);  //may need a sync in betweeen as sfShm has been updated after last barrier
    __syncthreads();
    
    //warps multiply tiles of S and V
    tcMatmul(sShm, vShm, oShm, tile_size, emb_dim, tile_size, tile_size);
  }
  __syncthreads();
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) O[tbOffset + i] = oShm[i];
}

//====================== Sparse tc fused attn ================================
__global__ void sparseTcFusedAttnKernel(half* Q, half* K, half* V, half* O, int seq_len, int emb_dim, int tile_size, int* row_ptr, int* col_idx){
  extern __shared__ float sp_float_shm[];  // declare shared memory as float first for alignment
  //allocate float-type shared memory first
  float* lShm = sp_float_shm;
  float* l_prev = lShm + blockDim.x * 2 * tile_size;
  float* sfShm = l_prev + tile_size;
  //integer data
  int* col_idx_shm = reinterpret_cast<int*>(sfShm + tile_size*tile_size); 
  //allocate half-type shared memory after the float section
  half* sp_tc_shm = reinterpret_cast<half*>(col_idx_shm + seq_len);
  half* qShm = sp_tc_shm;
  half* kShm = qShm + tile_size * emb_dim;
  half* sShm = kShm + emb_dim * tile_size;
  half* vShm = sShm + tile_size * tile_size;
  half* oShm = vShm + tile_size * emb_dim;
  //oShm size = tile_size*emb_dim

  unsigned int tid = threadIdx.y*blockDim.x + threadIdx.x;
  unsigned int idx = blockDim.x*blockDim.y*blockIdx.x + tid;  //assumes linear block indexing
  unsigned int tbOffset = blockIdx.x*tile_size*emb_dim;
  int warpTileHeight = tile_size/blockDim.y;
  unsigned int warpQOffset = threadIdx.y*emb_dim*warpTileHeight;
  unsigned int warpSOffset = threadIdx.y*tile_size*warpTileHeight;

  //bringing dense col info for this block row into shm
  int csr_row_start = row_ptr[blockIdx.x];
  int csr_row_next = row_ptr[blockIdx.x+1];
  int csr_row_size = csr_row_next - csr_row_start;
  for(unsigned int i=tid; i<csr_row_size; i += blockDim.x*blockDim.y) col_idx_shm[i] = col_idx[csr_row_start+i];
  int col_idx_shm_i = 0;
  int csr_last_col = col_idx_shm[csr_row_size - 1];
  __syncthreads();
  //loading Q tile into shm
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) qShm[i] = Q[tbOffset+i];
  //initialize O
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) oShm[i] = __float2half(0.0);
  //initialize l_prev
  for(unsigned int i=tid; i<tile_size; i += blockDim.x*blockDim.y) l_prev[i] = 0.0;


  for(int iter=0; iter<=csr_last_col; iter++){
    //pruning block
    if(iter != col_idx_shm[col_idx_shm_i]) continue;

    unsigned int iterOffset = iter*tile_size*emb_dim;

    //loading K tile into shm transposed
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y){
      unsigned int row = i/emb_dim;
      unsigned int col = i%emb_dim;

      kShm[col*tile_size+row] = K[iterOffset + i];  //32-way banck conflict as of now
    }
    //loading V tile into shm
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) vShm[i] = V[iterOffset+i];
    //initializing S tile
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sShm[i] = __float2half(0.0);
    __syncthreads();

    //warps multiply tiles
    tcMatmul(qShm, kShm, sShm, tile_size, tile_size, emb_dim, tile_size);
    __syncthreads();

    //exponentiate tile
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sfShm[i] = __expf(__half2float(sShm[i]));
    //softmax reduction
    for(unsigned int i=tid; i<blockDim.x*2*tile_size; i += blockDim.x*blockDim.y) lShm[i] = 0.0;
    __syncthreads();
    for(unsigned int row = threadIdx.y; row<tile_size; row += blockDim.y) warpReduce(sfShm+row*tile_size, lShm+row*blockDim.x*2, tile_size);
    __syncthreads();
    //applying corrected softmax
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y){
      int row = i/tile_size;
      sfShm[i] = sfShm[i]/(lShm[row*2*blockDim.x]+l_prev[row]); //first element of resp. row in l block (where elements have been reduced into)
    }

    //rescaling O
    for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y){
      int row = i/emb_dim;
      oShm[i] = __float2half((__half2float(oShm[i])*l_prev[row])/(lShm[row*2*blockDim.x]+l_prev[row]));
    }
    //update l_prev
    for(unsigned int i = tid; i<tile_size; i += blockDim.x*blockDim.y) l_prev[i] += lShm[i*2*blockDim.x];
    //downscaling from float to half
    for(unsigned int i=tid; i<tile_size*tile_size; i += blockDim.x*blockDim.y) sShm[i] = __float2half(sfShm[i]);  //may need a sync in betweeen as sfShm has been updated after last barrier
    __syncthreads();
    
    //warps multiply tiles of S and V
    tcMatmul(sShm, vShm, oShm, tile_size, emb_dim, tile_size, tile_size);

    //updating col_idx_shm ptr
    col_idx_shm_i++;
    __syncthreads();
  }
  __syncthreads();
  for(unsigned int i=tid; i<tile_size*emb_dim; i += blockDim.x*blockDim.y) O[tbOffset + i] = oShm[i];
}

__global__ void matTransKernel(float* AT, float* A, int N)  {
  int tid = blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
  for(int i = tid; i < N*N; i += blockDim.x*gridDim.x*blockDim.y) {
        int row = i / N;
        int col = i % N;
        AT[col*N+row] = A[i];
  }
}


//==========================================naive==========================================
// Kernel for matrix multiplication (QK^T)
__global__ void matmulKernel(float* output, const float* Q, const float* K, int N, int D) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
      float sum = 0.0;
      for (int i = 0; i < D; i++) {
          sum += Q[row * D + i] * K[col * D + i];  // Note: col * D instead of row * D for K^T
      }
      output[row * N + col] = sum;
  }
}

// Kernel for softmax computation
__global__ void softmaxKernel(float* scores, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N) {
      float maxVal = -INFINITY;
      
      // Compute row-wise max for numerical stability
      for (int i = 0; i < N; i++) {
          maxVal = fmaxf(maxVal, scores[row * N + i]);
      }

      float sum = 0.0;
      for (int i = 0; i < N; i++) {
          scores[row * N + i] = expf(scores[row * N + i] - maxVal);  // Apply stability trick
          sum += scores[row * N + i];
      }

      for (int i = 0; i < N; i++) {
          scores[row * N + i] /= sum;  // Normalize
      }
  }
}

// Kernel for weighted sum (Softmax * V)
__global__ void weightedSumKernel(float* output, const float* scores, const float* V, int N, int D) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < D) {
      float sum = 0.0;
      for (int i = 0; i < N; i++) {
          sum += scores[row * N + i] * V[i * D + col];
      }
      output[row * D + col] = sum;
  }
}
void naiveAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O) {
  int N = Q.size(0);   // Sequence length
  int D = Q.size(1);   // Embedding dimension
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());

  // Allocate tensor for scores
  torch::Tensor scores = torch::zeros({N, N}, options);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

  // Compute QK^T
  matmulKernel<<<numBlocks, threadsPerBlock>>>(scores.data_ptr<float>(), Q.data_ptr<float>(), K.data_ptr<float>(), N, D);
  cudaDeviceSynchronize();

  // Apply softmax
  softmaxKernel<<<(N + 255) / 256, 256>>>(scores.data_ptr<float>(), N);
  cudaDeviceSynchronize();

  // Compute weighted sum (Softmax * V)
  dim3 numBlocksV((D + 15) / 16, (N + 15) / 16);
  weightedSumKernel<<<numBlocksV, threadsPerBlock>>>(O.data_ptr<float>(), scores.data_ptr<float>(), V.data_ptr<float>(), N, D);
  cudaDeviceSynchronize();
}