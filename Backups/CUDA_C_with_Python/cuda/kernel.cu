#include <cuda.h>
#include <cuda_runtime_api.h>
#include<stdio.h>

__global__ void cuda_gray_kernel(unsigned char *b, unsigned char *g, unsigned char *r, unsigned char *gray, size_t size)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) {
		return;
	}

	gray[idx] = (unsigned char)(0.114f*b[idx] + 0.587f*g[idx] + 0.299f*r[idx] + 0.5);

}


extern "C"  {
	__declspec(dllexport) void cuda_gray(unsigned char *a, unsigned char *b, unsigned char *c, unsigned char *d, size_t size)
	{

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		unsigned char *d_a, *d_b, *d_c, *d_d;

		cudaMalloc((void **)&d_a, size * sizeof(char));
		cudaMalloc((void **)&d_b, size * sizeof(char));
		cudaMalloc((void **)&d_c, size * sizeof(char));
		cudaMalloc((void **)&d_d, size * sizeof(char));

		cudaMemcpy(d_a, a, size * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, c, size * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_d, d, size * sizeof(char), cudaMemcpyHostToDevice);


		cudaEventRecord(start);
		cuda_gray_kernel << < ceil(size / 1024.0), 1024 >> > (d_a, d_b, d_c, d_d, size);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Time on GPU : %f msec\n", milliseconds);

		cudaMemcpy(d, d_d, size * sizeof(char), cudaMemcpyDeviceToHost);

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		cudaFree(d_d);
	}


}
