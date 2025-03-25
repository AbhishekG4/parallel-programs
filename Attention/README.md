This is an extension to PyTorch with 4 custom kernels to perform the attention operation. The kernels increase in complexity from a naive implementation to one with support for block sparsity and tensor cores with the objective to be able to compare performance gain over each upgrade.

Future work:

- still more scope for optimizations.
- code clean up.
