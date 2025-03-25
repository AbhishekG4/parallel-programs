
#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <chrono>

void naiveAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O);
void matTrans(torch::Tensor AT, torch::Tensor A);
void fusedAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int seq_len, int emb_dim, int tile_size);
void tcFusedAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int seq_len, int emb_dim, int tile_size);
void sparseTcFusedAttn(torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O, int seq_len, int emb_dim, int tile_size, torch::Tensor row_ptr, torch::Tensor col_idx);

struct CSRMatrix {
  torch::Tensor row_ptr;
  torch::Tensor col_idx;
  //height and width assumed to be tile_size
  // Constructor
  CSRMatrix(torch::Tensor row_ptr, torch::Tensor col_idx)
      : row_ptr(std::move(row_ptr)), col_idx(std::move(col_idx)) {}
};

torch::Tensor naiveAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
  torch::Tensor O = torch::zeros_like(Q, torch::TensorOptions().device(Q.device()));
  naiveAttn(Q, K, V, O);
  
  return O;
}
torch::Tensor transpose(torch::Tensor A) {
  torch::Tensor AT = torch::zeros_like(A, torch::TensorOptions().device(A.device())); 
  matTrans(AT, A);
  return AT;
}
torch::Tensor fusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
  torch::Tensor O = torch::zeros_like(Q, torch::TensorOptions().device(Q.device()));
  fusedAttn(Q,K,V,O, Q.size(0),Q.size(1),32);
  return O;
}
torch::Tensor tcFusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
  torch::Tensor O = torch::zeros_like(Q, torch::TensorOptions().dtype(torch::kHalf).device(Q.device()));
  tcFusedAttn(Q.to(torch::kHalf),K.to(torch::kHalf),V.to(torch::kHalf),O, Q.size(0),Q.size(1),32);
  return O;
}
torch::Tensor sparseTcFusedAttention(torch::Tensor Q, torch::Tensor K, torch::Tensor V, CSRMatrix mask){
  torch::Tensor O = torch::zeros_like(Q, torch::TensorOptions().dtype(torch::kHalf).device(Q.device()));
  sparseTcFusedAttn(Q.to(torch::kHalf),K.to(torch::kHalf),V.to(torch::kHalf),O, Q.size(0),Q.size(1),32, mask.row_ptr, mask.col_idx);
  return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<CSRMatrix>(m, "CSRMatrix")
        .def(pybind11::init<torch::Tensor, torch::Tensor>()) // Constructor
        .def_readwrite("row_ptr", &CSRMatrix::row_ptr)  // Expose row_ptr
        .def_readwrite("col_idx", &CSRMatrix::col_idx); // Expose col_idx
  // function bindings:
  m.def("naive_transpose", &transpose, "naive transpose");
  // below are the functions you need to implement and compare
  m.def("naive_attention", &naiveAttention, "naive attention");
  m.def("fused_attention", &fusedAttention, "fused attention");
  m.def("tc_fused_attention", &tcFusedAttention, "fused attention with tensor cores");
  m.def("sparse_tc_fused_attention", &sparseTcFusedAttention, "sparse fused attention with tensor cores");
  // add more here if you have more variants to test
}