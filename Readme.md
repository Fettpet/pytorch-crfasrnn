# CRFasRNN
The CRFasRNN Model was introduced by Zheng el al [1]. The Source code [2] from Miguel Monteiro was changed, such that it can be used with pytorch.

# Install
To use the package, it's necessary to install the Permuthodral lattice. We recommand the usage of virtual enviourments. 

1. (Optional) Create a virtual enviourments: **python3 -m virtualenv /path/to/ENV/**
2. Activate the virtual enviourments: **source /path/to/ENV/bin/activate**
3. Goto the Permuthoderal dir: **cd /path/to/pytorch-crfasrnn/Permutohedral_Filtering**
4. install: **python setup.py install**
5. use it

# Usage 
After you install it, you can use the CRFasRNN.py. It has implemented the model like in Zheng. 

# change the number of segments
For performance reasons the source uses shared memory. The size of the memory depends on the number of segments. The size is given by a compile time template paramater. If you like to change the number of segments, you need to change this number. Sorry that you need to do this. But it is quite simple. You need to add an additional litticeFilterGpu.
1. Run it. Notice the pd and vd value from the error message
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

# Sources 
[1] https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf
[2] https://github.com/MiguelMonteiro/CRFasRNNLayer
