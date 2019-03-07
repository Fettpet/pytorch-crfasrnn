/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis, Sebastian Hahn

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

//#include "LatticeFilterKernel.h"

#include <torch/extension.h>

#include "PermutohedralLatticeGPU.cuh"
#include "DeviceMemoryAllocator.h"
#include "../Devices.hpp"
#ifndef SPATIAL_DIMS
#define SPATIAL_DIMS 2
#endif
#ifndef INPUT_CHANNELS
#define INPUT_CHANNELS 3
#endif
#ifndef REFERENCE_CHANNELS
#define REFERENCE_CHANNELS 3
#endif
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void computeKernelGPU(
    at::Tensor const & input_image,
    at::Tensor & positions,
    int num_super_pixels,
    int n_spatial_dims,
    int *spatial_dims,
    int n_reference_channels,
    float spatial_std,
    float features_std
)
{
    CHECK_INPUT(input_image);
    CHECK_INPUT(positions);

    int* spatial_dims_gpu;
    gpuErrchk(
        cudaMalloc(
            (void**)&spatial_dims_gpu,
            n_spatial_dims*sizeof(int)
        )
    );

    gpuErrchk(
        cudaMemcpy(
            spatial_dims_gpu,
            spatial_dims,
            n_spatial_dims*sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //return input_tensor;
    compute_kernel<float><<<blocks, blockSize>>>(
        input_image.data<float>(),
        positions.data<float>(),
        num_super_pixels,
        n_reference_channels,
        n_spatial_dims,
        spatial_dims_gpu,
        spatial_std,
        features_std
    );

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(
        cudaFree(
            spatial_dims_gpu
        )
    );
};

//declaration of what lattices (pd and vd) can be used


// Define the GPU implementation that launches the CUDA kernel.
template<
    int pd,
    int vd
>
void
latticeFilterGPU
(
    at::Tensor & output,
    at::Tensor const & input,
    at::Tensor const & positions,
    int num_super_pixels,
    bool reverse
)
{
    auto allocator = DeviceMemoryAllocator();
    //bilateral

    auto lattice = PermutohedralLatticeGPU<
        float,
        pd,
        vd
    >
    (
        num_super_pixels,
        &allocator
    );

    lattice.filter(
        output,
        input,
        positions,
        reverse
    );
}

at::Tensor
LatticeFilter_calculate_gpu(
    at::Tensor const & input_tensor,
    at::Tensor const & image_tensor,
    bool bilateral,
    float theta_alpha=1.0,
    float theta_beta=1.0,
    float theta_gamma=1.0,
    bool backward=false
)
{

    // calculate dimensions; dimension 0 is batch; dim 1 is channel
    int rank = image_tensor.ndimension();
    int n_spatial_dims = rank - 2;
    int pd;

    auto batch_size = static_cast<int>(input_tensor.size(0));
    auto n_input_channels = static_cast<int>(input_tensor.size(1));
    auto spatial_dims = new int[n_spatial_dims];

    int num_super_pixels{1};
    for (int i = 0; i < n_spatial_dims; i++){
        auto dim_size = static_cast<int>(image_tensor.size(i + 2)); // ignore the first two channels (batch and color)
        num_super_pixels *= dim_size;
        spatial_dims[i] = dim_size;
    }


    int vd = n_input_channels + 1;
    float spatial_std;
    float features_std;
    int n_reference_channels;

    if(bilateral){
        assert(image_tensor.dim() == rank);
        n_reference_channels = static_cast<int>(image_tensor.size(1));
        pd = n_reference_channels + n_spatial_dims;
        spatial_std = theta_alpha;
        features_std = theta_beta;
    }
    else
    {
        pd = n_spatial_dims;
        n_reference_channels = 0; //set to zero so ComputeKernel does not use reference image channels
        spatial_std = theta_gamma;
        features_std = -1; //does not matter
    }

    // Allocate kernel positions and calculate them

    at::Tensor positions = at::zeros(
        {batch_size * num_super_pixels * pd},
        input_tensor.type()
    );
    at::Tensor output_tensor = at::zeros_like(input_tensor);

    auto in_ptr = input_tensor.data<float>();

    auto out_ptr = output_tensor.data<float>();

    auto pos_ptr = positions.data<float>();


    computeKernelGPU(
        input_tensor,
        positions,
        num_super_pixels,
        n_spatial_dims,
        spatial_dims,
        n_reference_channels,
        spatial_std,
        features_std
    );

    if( pd==2 and vd==27)
    {
        latticeFilterGPU<
            2,
            27
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==5 and vd==27)
    {
        latticeFilterGPU<
            5,
            27
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==2 and vd==4)
    {
        latticeFilterGPU<
            2,
            4
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==5 and vd==4)
    {
        latticeFilterGPU<
            5,
            4
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==2 and vd==35)
    {
        latticeFilterGPU<
            2,
            35
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==5 and vd==35)
    {
        latticeFilterGPU<
            5,
            35
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==2 and vd==6)
    {
        latticeFilterGPU<
            2,
            6
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==3 and vd==6)
    {
        latticeFilterGPU<
            3,
            6
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else if( pd==5 and vd==6)
    {
        latticeFilterGPU<
            5,
            6
        >
        (
            output_tensor,
            input_tensor,
            positions,
            num_super_pixels,
            backward
        );
    }
    else
    {
            /**
            Sorry that you need to do this. But it is quite simple. You need to add an additional litticeFilterGpu.
            1. notice the pd and vd value from the error message
            2. add an else if(pd == ? and vd == ?)
            3. copy the following text. Replace vd and pd with the values
                latticeFilterGPU<
                    pd,
                    vd
                >
                (
                    output_tensor,
                    input_tensor,
                    positions,
                    num_super_pixels,
                    backward
                );
            4. recomplie  with setup.py
            */
        std::cerr << "latticeFilterGPU with pd=" << pd << " and vd=" << vd << " doesnt exists. Pls add this in"
                  << " the file latticeFilter.cu. This is nessacary for an efficent GPU implementation" <<std::endl;
        exit(1);
    }


    delete[](spatial_dims);
    return output_tensor;
}



at::Tensor
LatticeFilter_forward(
    at::Tensor const & input_tensor,
    at::Tensor const & image_tensor,
    bool bilateral,
    float theta_alpha=1.0,
    float theta_beta=1.0,
    float theta_gamma=1.0
)
{
    return LatticeFilter_calculate_gpu(
        input_tensor,
        image_tensor,
        bilateral,
        theta_alpha,
        theta_beta,
        theta_gamma,
        false // forward
    );

}

at::Tensor
LatticeFilter_backward(
    at::Tensor const & input_tensor,
    at::Tensor const & image_tensor,
    bool bilateral,
    float theta_alpha=1.0,
    float theta_beta=1.0,
    float theta_gamma=1.0
)
{

    return LatticeFilter_calculate_gpu(
        input_tensor,
        image_tensor,
        bilateral,
        theta_alpha,
        theta_beta,
        theta_gamma,
        true // backward
    );
}

// bind it to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &LatticeFilter_forward, "LatticeFilter forward");
  m.def("backward", &LatticeFilter_backward, "LatticeFilter backward");
}
