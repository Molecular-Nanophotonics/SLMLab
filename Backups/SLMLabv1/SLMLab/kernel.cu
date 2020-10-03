/*
 - Add "cufft.lib;" to "Properties > Linker > Additional Dependencies"
 - Change "Properties > General > Configuration Type" to "Dynamic Libary (.dll)"
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdint.h>
#include <cufft.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#define BLOCK_SIZE 256	// should be a power of 2
#define SLM_SIZE 512

// KERNEL DECLARATIONS
__global__ void ReplaceAmplitudeSample_FFT(float *g_dAmplitude_f, cufftComplex *g_FFTo_cc, cufftComplex *g_FFTd_cc);
__global__ void ReplaceAmplitudeSLM_FFT(cufftComplex *g_SLM_cc, bool last_iteration, uint16_t *g_pSLM_uint16);
__global__ void p2c(cufftComplex *g_c, float *g_p, int M);

// GLOBAL DECLARATIONS
int slm_size, n_pixels, n_blocks_phi;
float *d_pSLM_f, *d_pSLMstart_f;
cufftHandle plan;
cufftComplex *d_SLM_cc, *d_FFTd_cc, *d_FFTo_cc;
int memsize_SLM_cc, memsize_pSLM_f, memsize_pSLM_uint16, memsize_Amplitude;
float *d_dAmplitude_f;
uint16_t  *d_pSLM_uint16;

// CONSTANT DEVICE MEMORY DECLARATIONS
__device__ __constant__ int c_n_pixels[1];


// allocate GPU memory
extern "C" __declspec(dllexport) int startCUDA(float *h_pSLMstart_f, int deviceID)
{
	slm_size = SLM_SIZE;
	n_pixels = slm_size * slm_size;
	n_blocks_phi = (n_pixels / BLOCK_SIZE + (n_pixels % BLOCK_SIZE == 0 ? 0 : 1));

	cudaMemcpyToSymbol(c_n_pixels, &n_pixels, sizeof(int), 0, cudaMemcpyHostToDevice);

	memsize_SLM_cc = n_pixels * sizeof(cufftComplex);
	memsize_pSLM_f = n_pixels * sizeof(float);
	memsize_pSLM_uint16 = n_pixels * sizeof(uint16_t);
	memsize_Amplitude = n_pixels * sizeof(float);

	cudaMalloc(&d_dAmplitude_f, memsize_Amplitude);

	cudaMalloc(&d_pSLM_f, memsize_pSLM_f);
	cudaMalloc(&d_pSLM_uint16, memsize_pSLM_uint16);
	cudaMemcpy(d_pSLM_f, h_pSLMstart_f, memsize_pSLM_f, cudaMemcpyHostToDevice);

	cudaMalloc(&d_SLM_cc, memsize_SLM_cc);
	cudaMalloc(&d_FFTd_cc, memsize_SLM_cc);
	cudaMalloc(&d_FFTo_cc, memsize_SLM_cc);

	cudaDeviceSynchronize();

	p2c <<< n_blocks_phi, BLOCK_SIZE >>> (d_SLM_cc, d_pSLM_f, n_pixels);

	cudaDeviceSynchronize();

	cufftPlan2d(&plan, slm_size, slm_size, CUFFT_C2C);

	return 0;
}


extern "C" __declspec(dllexport) int generatePhase(uint16_t *h_pSLM_uint16, float *h_dAmplitude, int n_iterations)
{
	cudaMemcpy(d_dAmplitude_f, h_dAmplitude, memsize_Amplitude, cudaMemcpyHostToDevice);
	cudaMemset(d_FFTd_cc, 0, memsize_SLM_cc);
	cudaDeviceSynchronize();

	for (int i = 0; i < n_iterations; i++)
	{
		// transform to sample plane
		cufftExecC2C(plan, d_SLM_cc, d_FFTo_cc, CUFFT_FORWARD);
		cudaDeviceSynchronize();
		// copy phase of d_FFTo_cc to d_FFTd_cc
		ReplaceAmplitudeSample_FFT <<< n_blocks_phi, BLOCK_SIZE >>> (d_dAmplitude_f, d_FFTo_cc, d_FFTd_cc);
		cudaDeviceSynchronize();
		// transform back to SLM plane
		cufftExecC2C(plan, d_FFTd_cc, d_SLM_cc, CUFFT_INVERSE);
		cudaDeviceSynchronize();
		// set amplitudes in d_SLM_cc to the laser amplitude profile
		ReplaceAmplitudeSLM_FFT <<< n_blocks_phi, BLOCK_SIZE >>> (d_SLM_cc, (i == (n_iterations - 1)), d_pSLM_uint16);
	}
	cudaMemcpy(h_pSLM_uint16, d_pSLM_uint16, memsize_pSLM_uint16, cudaMemcpyDeviceToHost);

	return 0;
}


extern "C" __declspec(dllexport) int stopCUDA()
{
	cudaFree(d_pSLM_f);
	cudaFree(d_pSLMstart_f);
	cudaFree(d_SLM_cc);
	cudaDeviceReset();

	return 0;
}


__global__ void ReplaceAmplitudeSample_FFT(float *g_dAmplitude_f, cufftComplex *g_FFTo_cc, cufftComplex *g_FFTd_cc)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < c_n_pixels[0])
	{
		cufftComplex pFFTo_cc = g_FFTo_cc[idx];
		float pFFTo_f = atan2f(pFFTo_cc.y, pFFTo_cc.x);

		float dAmplitide_f = g_dAmplitude_f[idx];

		g_FFTd_cc[idx].x = dAmplitide_f*cosf(pFFTo_f);
		g_FFTd_cc[idx].y = dAmplitide_f*sinf(pFFTo_f);
	}
	__syncthreads();
}


__global__ void ReplaceAmplitudeSLM_FFT(cufftComplex *g_SLM_cc, bool last_iteration, uint16_t *g_pSLM_uint16)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < c_n_pixels[0])
	{
		cufftComplex pSLM_cc = g_SLM_cc[idx];
		float pSLM_f = atan2f(pSLM_cc.y, pSLM_cc.x);

		if (last_iteration) // get the SLM phase as uint16_t [0, 65535]
		{
			g_pSLM_uint16[idx] = (uint16_t)floor((pSLM_f + M_PI)*65536.0f / (2.0f * M_PI));
		}
		g_SLM_cc[idx].x = 1.0f*cosf(pSLM_f);
		g_SLM_cc[idx].y = 1.0f*sinf(pSLM_f);
	}
	__syncthreads();
}


// calculate complex from phases
__global__ void p2c(cufftComplex *g_c, float *g_p, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
	{
		float pSLM_f = g_p[idx];
		g_c[idx].x = cosf(pSLM_f);
		g_c[idx].y = sinf(pSLM_f);
	}
	__syncthreads();
}