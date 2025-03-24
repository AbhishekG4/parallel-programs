from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fastAttention',
    ext_modules=[
        CUDAExtension('fastAttention', [
            'fastAttention.cpp',
            'fastAttention_kernels.cu',
        ],
        extra_compile_args={
            'nvcc': [
              '-O3',
              '-use_fast_math',
              '-ftz=true',
              '-prec-div=false',
              '-prec-sqrt=false',
              '-lineinfo',
              ]
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
