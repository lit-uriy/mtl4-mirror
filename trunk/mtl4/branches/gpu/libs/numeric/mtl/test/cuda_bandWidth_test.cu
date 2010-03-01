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
// 
//  Usage:
//  ./cuda_bandwidth_test 
 

// includes, system
#include <sys/time.h>
#include <iostream>

// defines, project
#define MEMCOPY_ITERATIONS  10
#define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M
#define DEFAULT_INCREMENT   (1 << 22)               //4 M
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M
//enums, project
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };

// declaration, forward
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, 
                        memcpyKind kind, int currentDevice);
float testDeviceToHostTransfer(unsigned int memSize);
float testHostToDeviceTransfer(unsigned int memSize);
float testDeviceToDeviceTransfer(unsigned int memSize);

// Program main
int
main(int argc, char** argv) 
{
   
    int start = DEFAULT_SIZE;
    int end = DEFAULT_SIZE;
    int startDevice = 0;
    int endDevice = 0;
    int increment = DEFAULT_INCREMENT;
          
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if( deviceCount == 0 )
    {
        std::cout<< "!!!No devices found!!!\n";
        return 1;
    } else {        
        startDevice = 0;
        endDevice = deviceCount-1;
    }
    std::cout<<  "\n!!!Bandwidth test computed on all devices !!!\n";    
   
    for( int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, currentDevice);
	std::cout<< "\nRunning on...... device " <<  currentDevice << ": " << deviceProp.name << "\n\n";
       
	testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment, HOST_TO_DEVICE, currentDevice);
	testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment, DEVICE_TO_HOST, currentDevice);
	testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment, DEVICE_TO_DEVICE, currentDevice);
    }
    std::cout<< "Test passed\n";

    return 0;
}

void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, 
                   memcpyKind kind, int currentDevice)
{
    unsigned int memSizes;
    float bandwidths = 0.0f;

    //print information for use
    switch(kind)
    {
      case DEVICE_TO_HOST:    std::cout<< "Device to Host Bandwidth for ";
        break;
    case HOST_TO_DEVICE:      std::cout<<  "Host to Device Bandwidth for ";
        break;
    case DEVICE_TO_DEVICE:    std::cout<< "Device to Device Bandwidth for";
        break;
    }
    std::cout<< "Pageable memory\n";
          
    cudaSetDevice(currentDevice);
    //run each of the copies
    memSizes= start;
	switch(kind)
	{
	case DEVICE_TO_HOST:    bandwidths= testDeviceToHostTransfer( memSizes );
	    break;
	case HOST_TO_DEVICE:    bandwidths= testHostToDeviceTransfer( memSizes );
	    break;
	case DEVICE_TO_DEVICE:  bandwidths= testDeviceToDeviceTransfer( memSizes );
	    break;
	}
     std::cout<< "Transfer Size (Bytes)\tBandwidth(MB/s)\n" ;
     std::cout<< memSizes << "\t\t" << bandwidths << "\n\n";
	
    // Complete the bandwidth computation on all the devices
    cudaThreadExit();
}


//!  test the bandwidth of a device to host memcopy of a specific size
float testDeviceToHostTransfer(unsigned int memSize)
{
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;
    cudaEvent_t start, stop;

    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    
    //allocate host memory
    //pageable memory mode - use malloc
    h_idata = (unsigned char *)malloc( memSize );
    h_odata = (unsigned char *)malloc( memSize );
    
    //initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
      h_idata[i] = (unsigned char) (i & 0xff);
    
    // allocate device memory
    unsigned char* d_idata;
    cudaMalloc( (void**) &d_idata, memSize);

    //initialize the device memory
    cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);

    //copy data from GPU to Host
    cudaEventRecord( start, 0 );
    for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
	cudaMemcpy( h_odata, d_idata, memSize, cudaMemcpyDeviceToHost);
        
    // make sure GPU has finished copying
    cudaEventRecord( stop, 0 );
    cudaThreadSynchronize();
    //get the the total elapsed time in ms
    cudaEventElapsedTime( &elapsedTimeInMs, start, stop );
      
    //calculate bandwidth in MB/s
    bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / 
                                        (elapsedTimeInMs * (float)(1 << 20));

    //clean up memory
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    
    return bandwidthInMBs;
}


//! test the bandwidth of a host to device memcopy of a specific size
float testHostToDeviceTransfer(unsigned int memSize)
{
   // unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;
   
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    //allocate host memory
    unsigned char *h_odata = NULL;
    //pageable memory mode - use malloc
    h_odata = (unsigned char *)malloc( memSize );
    
    unsigned char *h_cacheClear1 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
    unsigned char *h_cacheClear2 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
    //initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
      h_odata[i] = (unsigned char) (i & 0xff);
    
    for(unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
        h_cacheClear1[i] = (unsigned char) (i & 0xff);
        h_cacheClear2[i] = (unsigned char) (0xff - (i & 0xff));
    }

    //allocate device memory
    unsigned char* d_idata;
    cudaMalloc( (void**) &d_idata, memSize);

    cudaEventRecord( start, 0 );
    //copy host memory to device memory
    for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	cudaMemcpy( d_idata, h_odata, memSize, cudaMemcpyHostToDevice);
    
    cudaEventRecord( stop, 0 );
    cudaThreadSynchronize();
    //total elapsed time in ms
    cudaEventElapsedTime( &elapsedTimeInMs, start, stop );
    
    //calculate bandwidth in MB/s
    bandwidthInMBs = (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / 
                                        (elapsedTimeInMs * (float)(1 << 20));

    //clean up memory
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    free(h_odata); 
    free(h_cacheClear1);
    free(h_cacheClear2);
    cudaFree(d_idata);

    return bandwidthInMBs;
}

//! test the bandwidth of a device to device memcopy of a specific size
float  testDeviceToDeviceTransfer(unsigned int memSize)
{
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;
   
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    //allocate host memory
    unsigned char *h_idata = (unsigned char *)malloc( memSize );
    //initialize the host memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
      h_idata[i] = (unsigned char) (i & 0xff);
 
    //allocate device memory
    unsigned char *d_idata;
    cudaMalloc( (void**) &d_idata, memSize);
    unsigned char *d_odata;
    cudaMalloc( (void**) &d_odata, memSize);

    //initialize memory
    cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);

    //run the memcopy
    cudaEventRecord( start, 0 );
    for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
      cudaMemcpy( d_odata, d_idata, memSize, cudaMemcpyDeviceToDevice);
    
    cudaEventRecord( stop, 0 );
    //Since device to device memory copies are non-blocking,
    //cudaThreadSynchronize() is required in order to get
    //proper timing.
    cudaThreadSynchronize();

    //get the the total elapsed time in ms
    cudaEventElapsedTime( &elapsedTimeInMs, start, stop );

    //calculate bandwidth in MB/s
    bandwidthInMBs = 2.0f * (1e3f * memSize * (float)MEMCOPY_ITERATIONS) / 
                                        (elapsedTimeInMs * (float)(1 << 20));
		
    //clean up memory
    free(h_idata);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return bandwidthInMBs;
}
