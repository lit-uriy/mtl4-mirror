// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

//#include <stdio.h>
#include <iostream>


int main( int argc, char** argv) 
{
    std::cout<<"CUDA Device Query\n";

    int deviceCount;

    cudaGetDeviceCount(&deviceCount);

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
        std::cout<<"There is no device supporting CUDA\n";
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        if (dev == 0) {
		// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
	        if (deviceProp.major == 9999 && deviceProp.minor == 9999)
                std::cout<<"There is no device supporting CUDA.\n";
            else if (deviceCount == 1)
                std::cout<<"There is 1 device supporting CUDA\n";
            else
                std::cout<<"There are "<<deviceCount<<" devices supporting CUDA\n";
        }
        std::cout<<"\nDevice "<< dev <<": \""<<deviceProp.name <<"\"\n";
    #if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		std::cout<<"  CUDA Driver Version:                           "<< driverVersion/1000<< "." << driverVersion%100<<"\n";
		cudaRuntimeGetVersion(&runtimeVersion);
		std::cout<<"  CUDA Runtime Version:                          "<< runtimeVersion/1000<< "." << runtimeVersion%100<<"\n";
    #endif

        std::cout<<"  CUDA Capability Major revision number:         "<<deviceProp.major<<"\n";
        std::cout<<"  CUDA Capability Minor revision number:         "<<deviceProp.minor<<"\n";

		std::cout<<"  Total amount of global memory:                 "<< deviceProp.totalGlobalMem<<" bytes\n";
    #if CUDART_VERSION >= 2000
        std::cout<<"  Number of multiprocessors:                     "<<deviceProp.multiProcessorCount<<"\n";
        std::cout<<"  Number of cores:                               "<<8 * deviceProp.multiProcessorCount<<"\n";
    #endif
        std::cout<<"  Total amount of constant memory:               "<<deviceProp.totalConstMem<<" bytes\n"; 
        std::cout<<"  Total amount of shared memory per block:       "<<deviceProp.sharedMemPerBlock<<" bytes\n";
        std::cout<<"  Total number of registers available per block: "<<deviceProp.regsPerBlock<<"\n";
        std::cout<<"  Warp size:                                     "<<deviceProp.warpSize<<"\n";
        std::cout<<"  Maximum number of threads per block:           "<<deviceProp.maxThreadsPerBlock<<"\n";
        std::cout<<"  Maximum sizes of each dimension of a block:    "<<deviceProp.maxThreadsDim[0]<<" x "<<deviceProp.maxThreadsDim[1]<<" x "<<deviceProp.maxThreadsDim[2]<<"\n";
        std::cout<<"  Maximum sizes of each dimension of a grid:     "<<deviceProp.maxGridSize[0]<<" x "<<deviceProp.maxGridSize[1]<<" x "<<deviceProp.maxGridSize[2]<<"\n";
        std::cout<<"  Maximum memory pitch:                          "<<deviceProp.memPitch<<" bytes\n";
        std::cout<<"  Texture alignment:                             "<<deviceProp.textureAlignment<<" bytes\n";
        std::cout<<"  Clock rate:                                    "<<deviceProp.clockRate * 1e-6f<<" GHz\n";
    #if CUDART_VERSION >= 2000
        std::cout<<"  Concurrent copy and execution:                 "<<(deviceProp.deviceOverlap ? "Yes\n" : "No\n");
    #endif
    #if CUDART_VERSION >= 2020
        std::cout<<"  Run time limit on kernels:                     "<<(deviceProp.kernelExecTimeoutEnabled ? "Yes\n" : "No\n");
        std::cout<<"  Integrated:                                    "<<(deviceProp.integrated ? "Yes\n" : "No\n");
        std::cout<<"  Support host page-locked memory mapping:       "<<(deviceProp.canMapHostMemory ? "Yes\n" : "No\n");
        std::cout<<"  Compute mode:                                  "<<(deviceProp.computeMode == cudaComputeModeDefault ? "Default (multiple host threads can use this device simultaneously)\n" : deviceProp.computeMode == cudaComputeModeExclusive ? "Exclusive (only one host thread at a time can use this device)\n" :deviceProp.computeMode == cudaComputeModeProhibited ? "Prohibited (no host thread can use this device)\n" :  "Unknown\n");
	
	
    #endif
}
    std::cout<<"\n\n";

   return 0;
}

