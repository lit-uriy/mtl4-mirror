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

#ifndef MTL_CUDA_CONFIG_INCLUDE
#define MTL_CUDA_CONFIG_INCLUDE


namespace mtl { namespace cuda {

#ifdef MTL_CUDA_HOST_LIMIT
    const unsigned host_limit= MTL_CUDA_HOST_LIMIT;
#else
    const unsigned host_limit= 1024;
#endif

template <typename T>
bool inline in_limit(const T& x) { return size(x) <= cuda::host_limit; }


// It takes the GPU with the highest number of cores as the best GPU. 

void activate_best_gpu(void)
{
  cudaDeviceProp device_properties;
  int max_cores = 0, 
      best_gpu = 0,  
      number_of_devices, 
      device_number;

  cudaError_t error;
      
  cudaGetDeviceCount(&number_of_devices);
  if (number_of_devices > 1) {
    for (device_number = 0; device_number < number_of_devices; device_number++) {
      cudaGetDeviceProperties(&device_properties, device_number);
      if (max_cores < device_properties.multiProcessorCount) {
	max_cores = device_properties.multiProcessorCount;
	best_gpu = device_number;
      }
    }
   
   }

    error=cudaSetDevice(best_gpu);
    if(error!=0){ 
	std::cout<<"\n==Error selecting GPU==\n"<<cudaGetErrorString(error) <<"\n\n";
	exit(1);
    } 

    else{
	cudaGetDeviceProperties(&device_properties, best_gpu);
	std::cout<< "\n\t\t===Running on device " <<  best_gpu << ": " << device_properties.name << "===\n\t\tNumber of Multiprocessors: "<<device_properties.multiProcessorCount<<"\n\n\n";
    } 

}



//  Activate a specific GPU
void activate_gpu(int currentDevice)
{
   cudaError_t error;
   error=cudaSetDevice(currentDevice);
   if(error!=0){ 
    std::cout<<"\n==Error selecting GPU==\n"<<cudaGetErrorString(error) <<"\n\n";
    exit(1);
   }else{   
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, currentDevice);
    std::cout<< "\n\t\t==Running on device " <<  currentDevice << ": " << deviceProp.name << "===\n\t\tNumber of Multiprocessors: "<<deviceProp.multiProcessorCount<<"\n\n\n";
   }

}



}} // namespace mtl::cuda

#endif // MTL_CUDA_CONFIG_INCLUDE
