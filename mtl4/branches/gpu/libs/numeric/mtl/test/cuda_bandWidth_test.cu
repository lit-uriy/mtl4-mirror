/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* 
 * This is a simple test program to measure the memcopy bandwidth of the GPU.
 * It can measure device to device copy bandwidth, host to device copy bandwidth 
 * for pageable and pinned memory, and device to host copy bandwidth for pageable 
 * and pinned memory.
 *
 * Usage:
 * ./bandwidthTest [option]...
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <iostream>

// includes, project
//#include <cutil_inline.h>
//#include <cuda.h>

// defines, project
#define MEMCOPY_ITERATIONS  10
#define DEFAULT_SIZE        ( 32 * ( 1 << 20 ) )    //32 M
#define DEFAULT_INCREMENT   (1 << 22)               //4 M
#define CACHE_CLEAR_SIZE    (1 << 24)               //16 M

//shmoo mode defines
#define SHMOO_MEMSIZE_MAX     (1 << 26)         //64 M
#define SHMOO_MEMSIZE_START   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_1KB   (1 << 10)         //1 KB
#define SHMOO_INCREMENT_2KB   (1 << 11)         //2 KB
#define SHMOO_INCREMENT_10KB  (10 * (1 << 10))  //10KB
#define SHMOO_INCREMENT_100KB (100 * (1 << 10)) //100 KB
#define SHMOO_INCREMENT_1MB   (1 << 20)         //1 MB
#define SHMOO_INCREMENT_2MB   (1 << 21)         //2 MB
#define SHMOO_INCREMENT_4MB   (1 << 22)         //4 MB
#define SHMOO_LIMIT_20KB      (20 * (1 << 10))  //20 KB
#define SHMOO_LIMIT_50KB      (50 * (1 << 10))  //50 KB
#define SHMOO_LIMIT_100KB     (100 * (1 << 10)) //100 KB
#define SHMOO_LIMIT_1MB       (1 << 20)         //1 MB
#define SHMOO_LIMIT_16MB      (1 << 24)         //16 MB
#define SHMOO_LIMIT_32MB      (1 << 25)         //32 MB

//enums, project
enum memcpyKind { DEVICE_TO_HOST, HOST_TO_DEVICE, DEVICE_TO_DEVICE };
enum printMode { USER_READABLE, CSV };
enum memoryMode { PINNED, PAGEABLE };

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(const int argc, const char **argv);
void testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, 
                        memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc);
float testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc);
float testDeviceToDeviceTransfer(unsigned int memSize);
void printResultsReadable(unsigned int *memSizes, float *bandwidths, unsigned int count);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv) 
{
    runTest(argc, (const char**)argv);
}

///////////////////////////////////////////////////////////////////////////////
//Parse args, run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
void runTest(const int argc, const char **argv)
{
    int start = DEFAULT_SIZE;
    int end = DEFAULT_SIZE;
    int startDevice = 0;
    int endDevice = 0;
    int increment = DEFAULT_INCREMENT;
    bool htod = true;
    bool dtoh = true;
    bool dtod = true;
    bool wc = false;
    printMode printmode = USER_READABLE;
    memoryMode memMode = PAGEABLE;
   
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if( deviceCount == 0 )
    {
        std::cout<< "!!!No devices found!!!\n";
        return;
    } else {
        std::cout<<  "!!!Cumulative Bandwidth to be computed from all the devices !!!\n\n";
        startDevice = 0;
        endDevice = deviceCount-1;
    }
         
    printf("Running on......\n");
    for( int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, currentDevice);
	std::cout<< "\n\n      device " <<  currentDevice << ": " << deviceProp.name << "\n";
       
	if( htod )
	    testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment, HOST_TO_DEVICE, printmode, memMode, startDevice, endDevice, wc);
	if( dtoh )
	    testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment, DEVICE_TO_HOST, printmode, memMode, startDevice, endDevice, wc);
	if( dtod )
	    testBandwidthRange((unsigned int)start, (unsigned int)end, (unsigned int)increment, DEVICE_TO_DEVICE, printmode, memMode, startDevice, endDevice, wc);
    }
    std::cout<< "Test passed\n";

    return;
}

///////////////////////////////////////////////////////////////////////
//  Run a range mode bandwidth test
//////////////////////////////////////////////////////////////////////
void
testBandwidthRange(unsigned int start, unsigned int end, unsigned int increment, 
                   memcpyKind kind, printMode printmode, memoryMode memMode, int startDevice, int endDevice, bool wc)
{
    //count the number of copies we're going to run
    unsigned int count = 1 + ((end - start) / increment);
    
    unsigned int *memSizes = ( unsigned int * )malloc( count * sizeof( unsigned int ) );
    float *bandwidths = ( float * ) malloc( count * sizeof(float) );

    //print information for use
    switch(kind)
    {
      case DEVICE_TO_HOST:    std::cout<< "Device to Host Bandwidth for ";
        break;
    case HOST_TO_DEVICE:      std::cout<<  "Host to Device Bandwidth for ";
        break;
    case DEVICE_TO_DEVICE:    std::cout<< "Device to Device Bandwidth\n";
        break;
    }
    if( DEVICE_TO_DEVICE != kind )
    {   switch(memMode)
        {
        case PAGEABLE:  std::cout<< "Pageable memory\n";
            break;
        case PINNED:    std::cout<< "Pinned memory\n";
            break;
        }
    }

    // Before calculating the cumulative bandwidth, initialize bandwidths array to NULL
    for (int i = 0; i < count; i++)
        bandwidths[i] = 0.0f;

    // Use the device asked by the user
    for (int currentDevice = startDevice; currentDevice <= endDevice; currentDevice++)
    {
        cudaSetDevice(currentDevice);
	    //run each of the copies
	    for(unsigned int i = 0; i < count; i++)
	    {
		memSizes[i] = start + i * increment;
	        switch(kind)
	        {
	        case DEVICE_TO_HOST:    bandwidths[i] = testDeviceToHostTransfer( memSizes[i], memMode, wc );
	            break;
	        case HOST_TO_DEVICE:    bandwidths[i] = testHostToDeviceTransfer( memSizes[i], memMode, wc );
	            break;
	        case DEVICE_TO_DEVICE:  bandwidths[i] = testDeviceToDeviceTransfer( memSizes[i] );
	            break;
	        }
	        printf(".");
		}
		cudaThreadExit();
    } // Complete the bandwidth computation on all the devices
    printf("\n");

    //print results
   printResultsReadable(memSizes, bandwidths, count);
   
    //clean up
    free(memSizes);
    free(bandwidths);
}

///////////////////////////////////////////////////////////////////////////////
//  test the bandwidth of a device to host memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float
testDeviceToHostTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
   // unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;
    cudaEvent_t start, stop;

    
    
//     cutCreateTimer( &timer );
     cudaEventCreate( &start );
     cudaEventCreate( &stop );
    
    //allocate host memory
    if( PINNED == memMode )
    {
        //pinned memory mode - use special function to get OS-pinned memory
#if CUDART_VERSION >= 2020
	cudaHostAlloc( (void**)&h_idata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 );
	cudaHostAlloc( (void**)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 );
#else
	cudaMallocHost( (void**)&h_idata, memSize );
        cudaMallocHost( (void**)&h_odata, memSize );
#endif
    }
    else
    {
        //pageable memory mode - use malloc
        h_idata = (unsigned char *)malloc( memSize );
        h_odata = (unsigned char *)malloc( memSize );
    }
    //initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_idata[i] = (unsigned char) (i & 0xff);
    }

    // allocate device memory
    unsigned char* d_idata;
    cudaMalloc( (void**) &d_idata, memSize);

    //initialize the device memory
    cudaMemcpy( d_idata, h_idata, memSize, cudaMemcpyHostToDevice);

    //copy data from GPU to Host
    cudaEventRecord( start, 0 );
    if( PINNED == memMode )
    {
        for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
             cudaMemcpyAsync( h_odata, d_idata, memSize, cudaMemcpyDeviceToHost, 0);
    }
    else
    {
        for( unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++ )
	    cudaMemcpy( h_odata, d_idata, memSize, cudaMemcpyDeviceToHost);
        
    }
//     cutilSafeCall( cudaEventRecord( stop, 0 ) );

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
    
    if( PINNED == memMode )
    {
        cudaFreeHost(h_idata);
        cudaFreeHost(h_odata);
    }
    else
    {
        free(h_idata);
        free(h_odata);
    }
    cudaFree(d_idata);
    
    return bandwidthInMBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a host to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float
testHostToDeviceTransfer(unsigned int memSize, memoryMode memMode, bool wc)
{
   // unsigned int timer = 0;
    float elapsedTimeInMs = 0.0f;
    float bandwidthInMBs = 0.0f;
    cudaEvent_t start, stop;
   
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    //allocate host memory
    unsigned char *h_odata = NULL;
    if( PINNED == memMode )
    {
#if CUDART_VERSION >= 2020
        //pinned memory mode - use special function to get OS-pinned memory
        cudaHostAlloc( (void**)&h_odata, memSize, (wc) ? cudaHostAllocWriteCombined : 0 );
#else
        //pinned memory mode - use special function to get OS-pinned memory
        cudaMallocHost( (void**)&h_odata, memSize );
#endif
    }
    else
    {
        //pageable memory mode - use malloc
        h_odata = (unsigned char *)malloc( memSize );
    }
    unsigned char *h_cacheClear1 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
    unsigned char *h_cacheClear2 = (unsigned char *)malloc( CACHE_CLEAR_SIZE );
    //initialize the memory
    for(unsigned int i = 0; i < memSize/sizeof(unsigned char); i++)
    {
        h_odata[i] = (unsigned char) (i & 0xff);
    }
    for(unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++)
    {
        h_cacheClear1[i] = (unsigned char) (i & 0xff);
        h_cacheClear2[i] = (unsigned char) (0xff - (i & 0xff));
    }

    //allocate device memory
    unsigned char* d_idata;
    cudaMalloc( (void**) &d_idata, memSize);

    cudaEventRecord( start, 0 );
    //copy host memory to device memory
    if( PINNED == memMode )
    {
        for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	    cudaMemcpyAsync( d_idata, h_odata, memSize, cudaMemcpyHostToDevice, 0);
    } else {
        for(unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
	    cudaMemcpy( d_idata, h_odata, memSize, cudaMemcpyHostToDevice);
    }

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
    
    if( PINNED == memMode )
    {
        cudaFreeHost(h_odata);
    }
    else
    {
        free(h_odata);
    }
    free(h_cacheClear1);
    free(h_cacheClear2);
    cudaFree(d_idata);

    return bandwidthInMBs;
}

///////////////////////////////////////////////////////////////////////////////
//! test the bandwidth of a device to device memcopy of a specific size
///////////////////////////////////////////////////////////////////////////////
float
testDeviceToDeviceTransfer(unsigned int memSize)
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
    {
        h_idata[i] = (unsigned char) (i & 0xff);
    }

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
    {
        cudaMemcpy( d_odata, d_idata, memSize, cudaMemcpyDeviceToDevice);
    }
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
		
    std::cout<< "elapsedTimeInMs=" << elapsedTimeInMs << "\n";
    std::cout<< "1111elapsedTimeInMs=" << elapsedTimeInMs << "\n";
    //clean up memory
   
    free(h_idata);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_idata);
    cudaFree(d_odata);

    return bandwidthInMBs;
}

/////////////////////////////////////////////////////////
//print results in an easily read format
////////////////////////////////////////////////////////
void printResultsReadable(unsigned int *memSizes, float *bandwidths, unsigned int count)
{
    printf("Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    for(unsigned int i = 0; i < count; i++)
    {
        printf("%9u\t\t%.1f\n", memSizes[i], bandwidths[i]);
    }
    printf("\n");
    fflush(stdout);
}


