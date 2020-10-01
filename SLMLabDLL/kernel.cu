/*
////////////////////////////////////////////////////////////////////////////////

Phase Pattern Generation on CUDA Devices for Spatial Light Modulators

Copyright 2020, Martin Fränzl
martin.fraenzl@uni-leipzig.de

Licensed under the GNU General Public License, Version 3.0;
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://opensource.org/licenses/GPL-3.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

////////////////////////////////////////////////////////////////////////////////
The function "generatePhase" contains two different algorithms for hologram
generation. The last parameter in the function call selects which one to use:
0: Complex addition of "Lenses and Prisms", no optimization (3D)
1: Weighted Gerchberg-Saxton algorithm using Fast Fourier Transforms (2D)


The ... algorithm (...) is described in:
...

////////////////////////////////////////////////////////////////////////////////
Naming conventions for variables:
- The prefix indicates where data is located
-- In global functions:
			c = constant memory
			g = global memory
			s = shared memory
			no prefix = registers
 -- In host functions:
			c = constant memory
			h = host memory
			d = device memory
- The suffix indicates the data type, no suffix usually indicates an integer

////////////////////////////////////////////////////////////////////////////////
Setup in Visual Studio:
- Add "cufft.lib;" to "Properties > Linker > Additional Dependencies"
- Change "Properties > General > Configuration Type" to "Dynamic Libary (.dll)"
*/

////////////////////////////////////////////////////////////////////////////////
// INCLUDES AND DEFINES
////////////////////////////////////////////////////////////////////////////////
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

//#include <stdint.h>
#include <cufft.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define MAX_SPOTS 256   // Decrease if GPU keeps running out of memory

#define BLOCK_SIZE 256	// Should be a power of 2
#define SLM_SIZE 512

#if ((SLM_SIZE==16)||(SLM_SIZE==32)||(SLM_SIZE==64)||(SLM_SIZE==128)||(SLM_SIZE==256)||(SLM_SIZE==512)||(SLM_SIZE==1024)||(SLM_SIZE==2048))
#define SLMPOW2			// Use bitwise modulu operations if the SLM size is a power of 2
#endif

////////////////////////////////////////////////////////////////////////////////
// FUNCTION DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

inline int computeAndCopySpotData(float* h_I, float* x, float* y, float* z, int N_spots, int method);

__global__ void LensesAndPrisms(unsigned char* g_SLMuc);

__global__ void ReplaceAmpsSLM_FFT(float* g_aLaser, cufftComplex* g_cAmp, float* g_pSLMstart, bool getpSLM255, unsigned char* g_pSLM255_uc);
__global__ void ReplaceAmpsSpots_FFT(cufftComplex* g_cSpotAmp_cc, cufftComplex* g_cSpotAmpNew_cc, int iteration, float* g_Iobtained, float* g_weight, bool last_iteration);



////////////////////////////////////////////////////////////////////////////////
// DEBUG MACROS
////////////////////////////////////////////////////////////////////////////////
/*
#define M_CHECK_ERROR() mCheckError(__LINE__, __FILE__)
#define M_SAFE_CALL(errcode) mSafeCall(errcode, __LINE__, __FILE__)
#define M_CUFFT_SAFE_CALL(cuffterror) mCufftSafeCall(cuffterror, __LINE__, __FILE__)
inline void mSafeCall(cudaError_t status, int line, char *file);
inline void mCufftSafeCall(cufftResult_t status, int line, char *file);
inline void mCheckError(int line, char *file);
*/

////////////////////////////////////////////////////////////////////////////////
// GLOBAL DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

float* d_Iobtained;
float SLMsizef = (float)SLM_SIZE;
int n_blocks_Phi, memsize_SLM_f, memsize_SLMuc, memsize_spotsf, data_w, N_pixels; // , N_iterations_last;
float h_desiredAmp[MAX_SPOTS];
int h_spotIndex[MAX_SPOTS];

unsigned char* d_pSLM_uc; // The optimized phase pattern, unsigned char, the one sent to the SLM [0, 255]
cudaError_t status;

float* d_pSLMstart_f; // The initial phase pattern [-pi, pi]
float* d_weights, * d_desiredAmp;
cufftHandle plan;
cufftComplex* d_FFTo_cc, * d_FFTd_cc, * d_SLM_cc;
int memsize_SLMcc;
float* d_aLaserFFT; // !!! Not used.

////////////////////////////////////////////////////////////////////////////////
// CONSTANT DEVICE MEMORY DECLARATIONS
////////////////////////////////////////////////////////////////////////////////
__device__ __constant__ int c_data_w[1];
__device__ __constant__ float c_data_w_f[1];
__device__ __constant__ int c_half_w[1];
__device__ __constant__ float c_half_w_f[1];
__device__ __constant__ int c_N_pixels[1];
__device__ __constant__ float c_N_pixels_f[1];
__device__ __constant__ float c_SLMpitch_f[1];
__device__ __constant__ int c_log2data_w[1];
__device__ __constant__ float c_x[MAX_SPOTS];
__device__ __constant__ float c_y[MAX_SPOTS];
__device__ __constant__ float c_z[MAX_SPOTS];
__device__ __constant__ float c_desiredAmp[MAX_SPOTS];
__device__ __constant__ int c_spotIndex[MAX_SPOTS];
__device__ __constant__ int c_N_spots[1];

////////////////////////////////////////////////////////////////////////////////
// FUNCTION TO TALK TO SLM HARDWARE
////////////////////////////////////////////////////////////////////////////////

// ... 


////////////////////////////////////////////////////////////////////////////////
// PUBLIC DLL FUNCTIONS 
////////////////////////////////////////////////////////////////////////////////
//uint16_t *h_pSLM_uint16 ?
extern "C" __declspec(dllexport) int generatePhase(unsigned char* h_pSLM_uc, float* x_spots, float* y_spots, float* z_spots, float* I_spots, int N_spots, int N_iterations, int method) //float *h_Iobtained, 
{
	if (N_spots > MAX_SPOTS)
	{
		N_spots = MAX_SPOTS;
	}
	else if (N_spots < 3)
		method = 0;

	//memsize_spotsf = N_spots * sizeof(float); // Required?
	computeAndCopySpotData(I_spots, x_spots, y_spots, z_spots, N_spots, method);

	switch (method)
	{
	case 0:
		////////////////////////////////////////////////////////////////////
		// Generate phase using "Lenses and Prisms" algorithm
		////////////////////////////////////////////////////////////////////
		LensesAndPrisms << < n_blocks_Phi, BLOCK_SIZE >> > (d_pSLM_uc);
		//M_CHECK_ERROR();
		cudaDeviceSynchronize();
		//M_CHECK_ERROR();
		cudaMemcpy(h_pSLM_uc, d_pSLM_uc, memsize_SLMuc, cudaMemcpyDeviceToHost); //M_SAFE_CALL()
		/*
		if (saveI_b)
		{
			calculateIobtained << < N_spots, SLM_SIZE >> >(d_pSLM_uc, d_Iobtained);
			M_CHECK_ERROR();
			cudaDeviceSynchronize();
			M_SAFE_CALL(cudaMemcpy(h_Iobtained, d_Iobtained, N_spots * sizeof(float), cudaMemcpyDeviceToHost));
		}
		*/
		break;

	case 1:
		////////////////////////////////////////////////////////////////////
		// Generate phase using Fresnel propagation (3D)
		////////////////////////////////////////////////////////////////////

		// ...
		break;

	case 2:
		////////////////////////////////////////////////////////////////////
		// Generate phase using Fast Fourier Transform (2D)
		////////////////////////////////////////////////////////////////////
		cudaMemcpy(d_desiredAmp, h_desiredAmp, memsize_spotsf, cudaMemcpyHostToDevice); // M_SAFE_CALL()
		cudaMemset(d_FFTd_cc, 0, memsize_SLMcc); // M_SAFE_CALL()
		//M_CHECK_ERROR();
		cudaDeviceSynchronize();
		for (int l = 0; l < N_iterations; l++)
		{
			// Transform to trapping plane
			cufftExecC2C(plan, d_SLM_cc, d_FFTo_cc, CUFFT_FORWARD); // M_CUFFT_SAFE_CALL()
			cudaDeviceSynchronize();
			// Copy phases for spot indices in d_FFTo_cc to d_FFTd_cc
			ReplaceAmpsSpots_FFT << < 1, N_spots >> > (d_FFTo_cc, d_FFTd_cc, l, d_Iobtained, d_weights, (l == (N_iterations - 1)));
			//M_CHECK_ERROR();
			cudaDeviceSynchronize();
			//Transform back to SLM plane
			cufftExecC2C(plan, d_FFTd_cc, d_SLM_cc, CUFFT_INVERSE); // M_CUFFT_SAFE_CALL()
			cudaDeviceSynchronize();
			// Set amplitudes in d_SLM to the laser amplitude profile
			ReplaceAmpsSLM_FFT << < n_blocks_Phi, BLOCK_SIZE >> > (d_aLaserFFT, d_SLM_cc, d_pSLMstart_f, (l == (N_iterations - 1)), d_pSLM_uc); // !!! d_aLaserFFT not used.
			//M_CHECK_ERROR();

			cudaDeviceSynchronize();
		}
		/*
		if (saveI_b)
			M_SAFE_CALL(cudaMemcpy(h_Iobtained, d_Iobtained, N_spots*(N_iterations) * sizeof(float), cudaMemcpyDeviceToHost));
		else
			M_SAFE_CALL(cudaMemcpy(h_Iobtained, d_weights, N_spots*(N_iterations) * sizeof(float), cudaMemcpyDeviceToHost));
		*/


		cudaMemcpy(h_pSLM_uc, d_pSLM_uc, memsize_SLMuc, cudaMemcpyDeviceToHost); // M_SAFE_CALL()

		break;
	default:
		break;
	}


	// Handle CUDA errors
	status = cudaGetLastError();

	return method;
}


////////////////////////////////////////////////////////////////////////////////
// ALLOCATE GPU MEMORY
////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport) int startCUDA(float* h_pSLMstart, int deviceId)
{
	// Make sure GPU with desired deviceId exists, set deviceId to 0 if not
	//int deviceCount = 0;
	/*
	if (cudaGetDeviceCount(&deviceCount) != 0)
		AfxMessageBox("No CUDA compatible GPU found.");
	if (deviceId >= deviceCount)
	{
		AfxMessageBox("Invalid deviceId, GPU with deviceId 0 used");
		deviceId = 0;
	}
	*/
	cudaSetDevice(deviceId); // M_SAFE_CALL()
	//cudaDeviceProp deviceProp;
	//M_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, deviceId));
	//maxThreads_device = deviceProp.maxThreadsPerBlock;


	int maxIterations = 1000;

	data_w = SLM_SIZE;
	cudaMemcpyToSymbol(c_data_w, &data_w, sizeof(int), 0, cudaMemcpyHostToDevice);
	float data_w_f = (float)data_w;
	cudaMemcpyToSymbol(c_data_w_f, &data_w_f, sizeof(float), 0, cudaMemcpyHostToDevice);
	int half_w = (int)(data_w / 2);
	cudaMemcpyToSymbol(c_half_w, &half_w, sizeof(int), 0, cudaMemcpyHostToDevice);
	float half_w_f = (float)data_w / 2.0f;
	cudaMemcpyToSymbol(c_half_w_f, &half_w_f, sizeof(float), 0, cudaMemcpyHostToDevice);
	N_pixels = data_w * data_w;
	cudaMemcpyToSymbol(c_N_pixels, &N_pixels, sizeof(int), 0, cudaMemcpyHostToDevice);
	float N_pixels_f = (float)N_pixels;
	cudaMemcpyToSymbol(c_N_pixels_f, &N_pixels_f, sizeof(float), 0, cudaMemcpyHostToDevice);
	int logN = (int)(log2(data_w_f));
	cudaMemcpyToSymbol(c_log2data_w, &logN, sizeof(int), 0, cudaMemcpyHostToDevice);
	float SLMpitch_f = 1.0f / data_w_f;
	cudaMemcpyToSymbol(c_SLMpitch_f, &SLMpitch_f, sizeof(float), 0, cudaMemcpyHostToDevice);

	n_blocks_Phi = (N_pixels / BLOCK_SIZE + (N_pixels % BLOCK_SIZE == 0 ? 0 : 1));

	memsize_spotsf = MAX_SPOTS * sizeof(float);
	memsize_SLM_f = N_pixels * sizeof(float);
	memsize_SLMuc = N_pixels * sizeof(unsigned char);
	memsize_SLMcc = N_pixels * sizeof(cufftComplex);
	//N_iterations_last = 10;

	// Memory allocations for all methods
	//M_SAFE_CALL(cudaMalloc((void**)&d_x, memsize_spotsf));
	//M_SAFE_CALL(cudaMalloc((void**)&d_y, memsize_spotsf));
	//M_SAFE_CALL(cudaMalloc((void**)&d_z, memsize_spotsf));
	//M_SAFE_CALL(cudaMalloc((void**)&d_I, memsize_spotsf));
	cudaMalloc((void**)&d_desiredAmp, memsize_spotsf); // M_SAFE_CALL()
	cudaMalloc((void**)&d_weights, MAX_SPOTS * (maxIterations + 1) * sizeof(float)); // M_SAFE_CALL()
	cudaMalloc((void**)&d_Iobtained, MAX_SPOTS * maxIterations * sizeof(float)); // M_SAFE_CALL()

	//M_SAFE_CALL(cudaMalloc((void**)&d_obtainedPhase, memsize_spotsf));
	//M_SAFE_CALL(cudaMalloc((void**)&d_spotRe_f, memsize_spotsf));
	//M_SAFE_CALL(cudaMalloc((void**)&d_spotIm_f, memsize_spotsf));

	//int data_w_pow2 = pow(2, ceil(log((float)data_w) / log(2.0f)));
	//M_SAFE_CALL(cudaMalloc((void**)&d_pSLM_f, data_w_pow2*data_w_pow2 * sizeof(float)));//the size of d_pSLM_f must be a power of 2 for the summation algorithm to work
	//M_SAFE_CALL(cudaMemset(d_pSLM_f, 0, data_w_pow2*data_w_pow2 * sizeof(float)));

	cudaMalloc((void**)&d_pSLM_uc, memsize_SLMuc); // M_SAFE_CALL()
	cudaMalloc((void**)&d_pSLMstart_f, memsize_SLM_f); // M_SAFE_CALL()
	cudaMemset(d_pSLMstart_f, 0, N_pixels * sizeof(float)); // M_SAFE_CALL()

	//M_SAFE_CALL(cudaMemcpy(d_pSLM_f, h_pSLMstart, N_pixels * sizeof(float), cudaMemcpyHostToDevice));

	// Memory allocations for FFT based Gerchberg-Saxton algorithm 
	//M_SAFE_CALL(cudaMalloc((void**)&d_spot_index, MAX_SPOTS * sizeof(int)));
	cudaMalloc((void**)&d_FFTd_cc, memsize_SLMcc); // M_SAFE_CALL()
	cudaMalloc((void**)&d_FFTo_cc, memsize_SLMcc); // M_SAFE_CALL()
	cudaMalloc((void**)&d_SLM_cc, memsize_SLMcc); // M_SAFE_CALL()
	cudaDeviceSynchronize(); // M_SAFE_CALL()
	//p2c << < n_blocks_Phi, BLOCK_SIZE >> >(d_SLM_cc, d_pSLM_f, N_pixels);
	//M_CHECK_ERROR();
	cudaDeviceSynchronize();
	cufftPlan2d(&plan, data_w, data_w, CUFFT_C2C); // M_CUFFT_SAFE_CALL()
	//float *h_aLaserFFT = (float *)malloc(memsize_SLM_f); // !!! Not used.

	status = cudaGetLastError();

	return status;
}

////////////////////////////////////////////////////////////////////////////////
// Stop CUDA and free GPU memory
////////////////////////////////////////////////////////////////////////////////
extern "C" __declspec(dllexport) int stopCUDA()
{
	//M_SAFE_CALL(cudaFree(d_x));
	//M_SAFE_CALL(cudaFree(d_y));
	//M_SAFE_CALL(cudaFree(d_z));
	//M_SAFE_CALL(cudaFree(d_I));

	cudaFree(d_weights); // M_SAFE_CALL()
	cudaFree(d_Iobtained); // M_SAFE_CALL()
	//M_SAFE_CALL(cudaFree(d_pSLM_f));
	cudaFree(d_pSLMstart_f); // M_SAFE_CALL()
	cudaFree(d_pSLM_uc); // M_SAFE_CALL()

	cudaFree(d_FFTd_cc); // M_SAFE_CALL()
	cudaFree(d_FFTo_cc); // M_SAFE_CALL()
	cudaFree(d_SLM_cc); // M_SAFE_CALL()
	cufftDestroy(plan); // //M_CUFFT_SAFE_CALL()

	cudaDeviceReset();

	status = cudaGetLastError();
	return status;
}

////////////////////////////////////////////////////////////////////////////////
// DEVICE FUNCTIONS 
////////////////////////////////////////////////////////////////////////////////
__device__ unsigned char phase2uc(float phase2pi)
{
	return (unsigned char)floor((phase2pi + M_PI) * 256.0f / (2.0f * M_PI));
}

__device__ int getXint(int index)
{
#ifdef SLMPOW2
	int X_int = index & (c_data_w[0] - 1);
#else
	float X_int = index % c_data_w[0];
#endif	
	return X_int;
}

__device__ int getYint(int index, int X_int)
{
#ifdef SLMPOW2
	int Y_int = (index - X_int) >> c_log2data_w[0];
#else
	int Y_int = (float)(floor((float)index / c_data_w_f[0]));
#endif	
	return Y_int;
}

__device__ int fftshift(int idx, int X, int Y)
{
	if (X < c_half_w[0])

	{
		if (Y < c_half_w[0])
		{
			return idx + (c_data_w[0] * c_half_w[0]) + c_half_w[0];
		}
		else
		{
			return idx - (c_data_w[0] * c_half_w[0]) + c_half_w[0];
		}
	}
	else
	{
		if (Y < c_half_w[0])
		{
			return idx + (c_data_w[0] * c_half_w[0]) - c_half_w[0];
		}
		else
		{
			return idx - (c_data_w[0] * c_half_w[0]) - c_half_w[0];
		}
	}

}

inline int computeAndCopySpotData(float* h_I, float* x, float* y, float* z, int N_spots, int method)
{
	for (int j = 0; j < N_spots; j++)
	{
		float sincx_rec = (x[j] == 0) ? 1.0f : ((M_PI * x[j] / SLMsizef) / sinf(M_PI * x[j] / SLMsizef));
		float sincy_rec = (y[j] == 0) ? 1.0f : ((M_PI * y[j] / SLMsizef) / sinf(M_PI * y[j] / SLMsizef));
		h_desiredAmp[j] = (h_I[j] <= 0.0f) ? 1.0f : (sincx_rec * sincy_rec * sqrtf(h_I[j] / 100) * SLMsizef * SLMsizef);
		if (method == 2)
			h_spotIndex[j] = ((int)(x[j]) & (data_w - 1)) + ((int)(y[j]) & (data_w - 1)) * data_w;
	}
	cudaMemcpyToSymbol(c_x, x, N_spots * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_y, y, N_spots * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_z, z, N_spots * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_desiredAmp, h_desiredAmp, N_spots * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(c_N_spots, &N_spots, sizeof(int), 0, cudaMemcpyHostToDevice);
	if (method == 2)
		cudaMemcpyToSymbol(c_spotIndex, h_spotIndex, N_spots * sizeof(int), 0, cudaMemcpyHostToDevice);

	return method;
}

////////////////////////////////////////////////////////////////////////////////
// Calculate phase using the "Lenses and Prisms" algorithm
////////////////////////////////////////////////////////////////////////////////
__global__ void LensesAndPrisms(unsigned char* g_SLMuc)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;

	if (idx < c_N_pixels[0])
	{
		// Get pixel coordinates
		int X_int = getXint(idx);
		int Y_int = getYint(idx, X_int);
		float X = c_SLMpitch_f[0] * (X_int - c_half_w_f[0]);
		float Y = c_SLMpitch_f[0] * (Y_int - c_half_w_f[0]);

		float phase2pi;
		float SLMre = 0.0f;
		float SLMim = 0.0f;

		for (int ii = 0; ii < c_N_spots[0]; ++ii)
		{
			// Add variable phases to function call 
			phase2pi = M_PI * c_z[ii] * (X * X + Y * Y) + 2.0f * M_PI * (X * (c_x[ii]) + Y * (c_y[ii]));
			SLMre = SLMre + c_desiredAmp[ii] * cosf(phase2pi);
			SLMim = SLMim + c_desiredAmp[ii] * sinf(phase2pi);
		}
		phase2pi = atan2f(SLMim, SLMre);	// [-pi,pi]

		g_SLMuc[idx] = phase2uc(phase2pi);
	}
}


////////////////////////////////////////////////////////////////////////////////
// Functions weighted Gerchberg-Saxton algorithm using Fast Fourier Transforms
////////////////////////////////////////////////////////////////////////////////

// Compute the phase in SLM plane and set amplitude to unity or Laser amplitude
__global__ void ReplaceAmpsSLM_FFT(float* g_aLaser, cufftComplex* g_cAmp, float* g_pSLMstart, bool getpSLM255, unsigned char* g_pSLM255_uc)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < c_N_pixels[0])
	{
		float aLaser = 1.0f; // g_aLaser[idx];

		cufftComplex cAmp = g_cAmp[idx];
		float pSLM2pi_f = atan2f(cAmp.y, cAmp.x);

		if (getpSLM255)
		{
			int X_int = getXint(idx);
			int Y_int = getYint(idx, X_int);
			int shiftedidx = fftshift(idx, X_int, Y_int);
			g_pSLM255_uc[shiftedidx] = phase2uc(pSLM2pi_f);
		}

		g_cAmp[idx].x = aLaser * cosf(pSLM2pi_f);
		g_cAmp[idx].y = aLaser * sinf(pSLM2pi_f);
	}
	__syncthreads();
}

// Adjust amplitudes in target plane
__global__ void ReplaceAmpsSpots_FFT(cufftComplex* g_cSpotAmp_cc, cufftComplex* g_cSpotAmpNew_cc, int iteration, float* g_Iobtained, float* g_weight, bool last_iteration)
{
	int tid = threadIdx.x;
	int spotIndex;
	float pSpot;
	__shared__ float s_aSpot[MAX_SPOTS], s_ISpotsMeanSq;
	float weight;
	cufftComplex cSpotAmp_cc;

	if (tid < c_N_spots[0])
	{
		spotIndex = c_spotIndex[tid];
		cSpotAmp_cc = g_cSpotAmp_cc[spotIndex];
		pSpot = atan2f(cSpotAmp_cc.y, cSpotAmp_cc.x);
		s_aSpot[tid] = hypotf(cSpotAmp_cc.x, cSpotAmp_cc.y) / c_desiredAmp[tid];
		if (iteration != 0)
			weight = g_weight[tid + iteration * c_N_spots[0]];
		else
		{
			s_aSpot[tid] = (s_aSpot[tid] < 0.5f) ? 0.5f : s_aSpot[tid];
			weight = c_desiredAmp[tid];
		}
	}
	__syncthreads();

	// Compute weights 
	if (tid == 0)
	{
		float ISpot_sum = 0.0f;
		for (int jj = 0; jj < c_N_spots[0]; jj++)
		{
			ISpot_sum += s_aSpot[jj] * s_aSpot[jj];
		}
		s_ISpotsMeanSq = sqrtf(ISpot_sum / (float)c_N_spots[0]);
	}
	__syncthreads();
	if (tid < c_N_spots[0])
	{
		weight = weight * s_ISpotsMeanSq / s_aSpot[tid];
		cSpotAmp_cc.x = cosf(pSpot) * weight;
		cSpotAmp_cc.y = sinf(pSpot) * weight;
		g_cSpotAmpNew_cc[spotIndex] = cSpotAmp_cc;

		if (last_iteration)
			g_weight[tid] = weight;
		else
			g_weight[c_N_spots[0] * (iteration + 1) + tid] = weight;
		/*
		if (c_saveI_b[0])
			g_Iobtained[c_N_spots[0] * (iteration)+tid] = s_aSpot[tid] * s_aSpot[tid];
		*/
	}
}