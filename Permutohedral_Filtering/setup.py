from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

setup(
    name='Permutohedral',
    ext_modules=[
        CppExtension(
            'Permutohedral',
            ['source/cpu/LatticeFilterKernel.cpp']
        ),
        CUDAExtension(
            'Permutohedral_gpu',
            ['source/gpu/LatticeFilter.cu']
        )

    ],
    cmdclass={'build_ext': BuildExtension}
)
