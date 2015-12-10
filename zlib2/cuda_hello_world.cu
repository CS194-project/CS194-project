// This is the REAL "hello world" for CUDA!
// It takes the string "Hello ", prints it, then passes it to CUDA with an array
// of offsets. Then the offsets are added in parallel to produce the string "World!"
// By Ingemar Ragnemalm 2010
 
#include <stdio.h>
#include "cuda_hello_world.h"

const int N = 16; 
const int blocksize = 16; 
 
__global__ 
extern void hello_cuda(char *a, int *b) 
{
	a[threadIdx.x] += b[threadIdx.x];
}
 
extern void cuda_hello_world()
{
	char a[N] = {0};
	int b[N] = {int('H'), int('e'), int('l'), int('l'), int('o'), int(' '), int('W'), int('o'), int('r'), int('l'), int('d'), int('!')};
 
	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);
 
	cudaMalloc( (void**)&ad, csize ); 
	cudaMalloc( (void**)&bd, isize ); 
	cudaMemcpy( ad, a, csize, cudaMemcpyHostToDevice ); 
	cudaMemcpy( bd, b, isize, cudaMemcpyHostToDevice ); 
	
	dim3 dimBlock( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	hello_cuda<<<dimGrid, dimBlock>>>(ad, bd);
	cudaMemcpy( a, ad, csize, cudaMemcpyDeviceToHost ); 
	cudaFree( ad );
	cudaFree( bd );
	
	printf("%s\n", a);
}

