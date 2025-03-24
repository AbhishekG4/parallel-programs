import torch
import numpy as np
import fastAttention
import argparse
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def pytorch_vanilla_attention(Q, K, V, mask=None):
  warmup = 10
  niters = 20
  Q = Q.unsqueeze(0).unsqueeze(1)
  K = K.unsqueeze(0).unsqueeze(1)
  V = V.unsqueeze(0).unsqueeze(1)
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  with sdpa_kernel(backends=[SDPBackend.MATH]):
    for _ in range(warmup):
      ref_attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, scale=1.0, attn_mask=mask)
    start.record()
    for _ in range(niters):
      ref_attn = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0, scale=1.0, attn_mask=mask)
    end.record()
  end.synchronize()
  print(f"Vanilla attention time: {start.elapsed_time(end)/10} ms")
  return ref_attn

def main(args):
  embed_dim = args.embed_dim
  seq_len = args.seq_len
  Q = torch.randn(args.seq_len, args.embed_dim, 
                  dtype=torch.float32, device="cuda")
  K = torch.randn(args.seq_len, args.embed_dim, 
                  dtype=torch.float32, device="cuda")
  V = torch.randn(args.seq_len, args.embed_dim, 
                  dtype=torch.float32, device="cuda")

  #Q = 0.1*Q
  ref_attn = pytorch_vanilla_attention(Q, K, V).squeeze(0).squeeze(0)

  my_attn = fastAttention.naive_attention(Q, K, V)
  # S = torch.matmul(Q,K.T)
  # ref_attn = torch.matmul(S,V)
  fus_attn = fastAttention.fused_attention(Q, K, V)
  print("=====ref=====")
  print(ref_attn)
  print("=====naive=====")
  print(my_attn)
  print("=====fused=====")
  print(fus_attn)
  tc_attn = fastAttention.tc_fused_attention(Q, K, V)
  print("=====tc======")
  print(tc_attn)
  # check relative error instead of absolute error
  # assert torch.norm(ref_attn - tc_attn)/torch.norm(ref_attn) < 1e-3, "Attention is incorrect"

  # these following parameters are just examples
  block_height = 32  #tile_size
  block_width = 32   #tile_size
  # seq_len = 8
  sparsity = 0.5
  mask = gen_mask(seq_len, sparsity, block_height, block_width)
  dense_mask = block_sparse_mast_to_dense(mask, seq_len, block_height, block_width)
  torch_sparse_attn = pytorch_vanilla_attention(Q, K, V, dense_mask)
  # print(mask.row_ptr)
  # print(mask.col_idx)#-----------
  mask.row_ptr = mask.row_ptr.to("cuda")
  mask.col_idx = mask.col_idx.to("cuda")
  mask = fastAttention.CSRMatrix(mask.row_ptr, mask.col_idx)
  sp_attn = fastAttention.sparse_tc_fused_attention(Q, K, V, mask)
  print("=====ref=====")
  print(torch_sparse_attn)
  print("=====sp=====")
  print(sp_attn)

class CSRMatrix:
  def __init__(self, row_ptr, col_idx, height, width):
    self.row_ptr = row_ptr
    self.col_idx = col_idx
    self.height = height
    self.width = width

# sparsity: ratio of non-zero blocks to total number of blocks
# block_height, block_width: size of the block. 
# This can be the block you assign to each thread block or each warp.
# generate block-sparse mask as a block-CSR matrix
# Instead of row_ptr tracking how many non-zero elements are in each row, 
# it tracks how many non-zero blocks are in each row.
# Instead of col_idx tracking the column index of each non-zero element, 
# it tracks the column index of each non-zero block.
# We also dont store the value of the non-zero blocks, as it is always 1.0
# Example:
# Sparse matrix of size 4x4, block_height = 2, block_width = 2
# M = [[1, 1, 0, 0],
#      [1, 1, 0, 0],
#      [0, 0, 1, 1],
#      [0, 0, 1, 1]]
# mask.row_ptr = [0, 1, 2]
# mask.col_idx = [0, 1]
def gen_mask(seq_len, sparsity, block_height, block_width):
  assert sparsity > 0.0, "Sparsity must be greater than 0"
  assert sparsity < 1.0, "Sparsity must be less than 1"
  assert seq_len >= block_height, "Sequence length must be greater than block height"
  assert seq_len >= block_width, "Sequence length must be greater than block width"
  assert seq_len < np.iinfo(np.int32).max, "Sequence length must be less than 2^31-1"

  n_block_per_row = int(np.ceil(seq_len / block_width))
  n_block_per_col = int(np.ceil(seq_len / block_height))
  num_blocks = n_block_per_row * n_block_per_col
  num_non_zero_blocks = int(num_blocks * sparsity)
  non_zero_blocks = np.random.choice(num_blocks, num_non_zero_blocks, replace=False)
  non_zero_blocks = np.sort(non_zero_blocks)
  row_indices = non_zero_blocks // n_block_per_row
  col_indices = non_zero_blocks % n_block_per_row
  row_ptr = np.zeros(n_block_per_col, dtype=np.int32)
  for row_idx in row_indices:
    row_ptr[row_idx] += 1
  row_ptr = np.cumsum(row_ptr)
  row_ptr = np.concatenate([[0], row_ptr])
  row_ptr = torch.IntTensor(row_ptr)
  col_indices = torch.IntTensor(col_indices)
  return CSRMatrix(row_ptr, col_indices, n_block_per_col, n_block_per_row)

# convert block-sparse mask to dense mask required by torch.scaled_dot_product_attention
def block_sparse_mast_to_dense(mask, seq_len, block_height, block_width):
  n_block_per_row = mask.row_ptr.size(0) - 1
  n_block_per_col = mask.col_idx.size(0)
  dense_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device="cuda")
  for i in range(n_block_per_row):
    for j in range(mask.row_ptr[i], mask.row_ptr[i+1]):
      col_idx = mask.col_idx[j]
      dense_mask[i*block_height:(i+1)*block_height, col_idx*block_width:(col_idx+1)*block_width] = True
  return dense_mask

if __name__ == "__main__":
  # test our transpose kernel
  # only for demonstration purposes
  Q = torch.randn(50, 50, device="cuda")
  QT = fastAttention.naive_transpose(Q)
  QT_ref = Q.T
  assert torch.allclose(QT, QT_ref), "Transpose kernel is incorrect"

  parser = argparse.ArgumentParser()
  parser.add_argument("--embed_dim","-e", type=int, default=128)
  parser.add_argument("--seq_len","-s", type=int, default=1024)
  args = parser.parse_args()

  main(args)

