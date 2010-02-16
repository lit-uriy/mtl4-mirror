//license

__global__ void vec_sum(int *out, int *in, unsigned int n, unsigned int blocksize)
{
        extern __shared__ int sdata[];

        //all threads load one element to shared memory
        unsigned int id= threadIdx.x;
        unsigned int i = blockIdx.x * (blocksize*2) + id;
        unsigned int gridSize = blocksize*2*gridDim.x;
        sdata[id]= 0;
        while (i < n){
                sdata[id]+= in[i] + in[i+blocksize];
                i += gridSize;
        }
        __syncthreads();

        //reduction in shared memory
        if (blocksize >= 512) {
                if (id < 256) sdata[id]+= sdata[id + 256];
                __syncthreads();
        }
        if (blocksize >= 256) {
                if (id < 128) sdata[id]+= sdata[id + 128];
                __syncthreads();
        }
        if (blocksize >= 128) {
                if (id < 64) sdata[id]+= sdata[id + 64];
                __syncthreads();
        }
        if (id < 32){
                if (blocksize >= 64) sdata[id]+= sdata[id + 32];
                if (blocksize >= 32) sdata[id]+= sdata[id + 16];
                if (blocksize >= 16) sdata[id]+= sdata[id +  8];
                if (blocksize >=  8) sdata[id]+= sdata[id +  4];
                if (blocksize >=  4) sdata[id]+= sdata[id +  2];
                if (blocksize >=  2) sdata[id]+= sdata[id +  1];
        }
        //write result of block to global memory
        if (id == 0) out[blockIdx.x]= sdata[0];

}

