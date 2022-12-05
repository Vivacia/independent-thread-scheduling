#include <cstdio>
#include <cstdlib>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

int mutex = 0;

__device__ long getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

__global__ void demo(dim3* threads, dim3* blocks, volatile int* mutex) {
    // locked = 1, unlocked = 0
    printf("here\n");
    while(atomicCAS((unsigned int*) mutex, (unsigned int) 0, (unsigned int) 1) == 1);
    long tid = getIdx(threads, blocks);
    printf("TID: %ld\n", tid);

    *mutex = (unsigned int) 0;
}

void demo_setup(dim3 threadsPerBlock, dim3 blocksPerGrid) {
    dim3* d_threads;
    dim3* d_blocks;
    checkCudaErrors(cudaMalloc((void**) &d_threads,
        sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &d_blocks,
        sizeof(dim3)));
    checkCudaErrors(cudaMemcpy(d_threads, &threadsPerBlock,
        sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_blocks, &blocksPerGrid,
        sizeof(dim3),cudaMemcpyHostToDevice));
    
    int* m;
    checkCudaErrors(cudaMalloc((int**) &m, sizeof(int)));
    checkCudaErrors(cudaMemcpy(m, &mutex, sizeof(int),
        cudaMemcpyHostToDevice));
    demo<<<blocksPerGrid, threadsPerBlock>>>(d_threads, d_blocks, m);
    checkCudaErrors(cudaMemcpy(&mutex, m, sizeof(int),
        cudaMemcpyDeviceToHost));
}

int main(int argc, char *argv[]) {
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 1;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    printf("before demo\n");
    demo_setup(threadsPerBlock, blocksPerGrid);
    printf("after demo\n");
}