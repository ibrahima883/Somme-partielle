
#include "PartialSum.h"


__global__ void PartialSum_Kernel(float *V, float *data, int nb_rows, int nb_columns)
{
	int j = threadIdx.x + blockIdx.x*blockDim.x, i;
	if (j < nb_columns) {
		for (i = 0; i < nb_rows; ++i)
			V[j] += data[i*nb_columns + j];
	}
}

// Helper function for using CUDA to add all rows of each columns in parallel.
cudaError_t addWithCuda(float *V, float *data, int nb_rows, int nb_columns)
{
	int size = nb_rows*nb_columns;
	float *dev_V, *dev_data;
	clock_t t0, t1;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	t0 = clock();
	// Allocate GPU buffers for two vectors (one input, one output)    .
	cudaStatus = cudaMalloc(&dev_data, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&dev_V, nb_columns * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	// Copy input vector from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_data, data, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_V, V, nb_columns * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	t1 = clock();
	cout << "  Allocation and Copy (CPU-->GPU) time on the GPU = " << (double)(t1 - t0) / CLOCKS_PER_SEC << " s" << endl;

	dim3 nThreadsPerBlock(SIZE_BLOCK_X, 1, 1);
	dim3 nBlocks((unsigned int)ceil(nb_columns / (float)SIZE_BLOCK_X), 1, 1);

	// Launch a kernel on the GPU with one thread for each column.
	PartialSum_Kernel<<< nBlocks, nThreadsPerBlock >>>(dev_V, dev_data, nb_rows, nb_columns);

	t0 = clock();
	cout << "  GPU time of Partial Sum = " << (double)(t0 - t1) / CLOCKS_PER_SEC << " s" << endl;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "PartialSum_Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching PartialSum_Kernel!\n", cudaStatus);
		goto Error;
	}
	
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(V, dev_V, nb_columns * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	t1 = clock();
	cout << "  Copy (GPU-->CPU) time of Partial Sum on the CPU = " << (double)(t1 - t0) / CLOCKS_PER_SEC << " s" << endl;

Error:
	cudaFree(dev_V);
	cudaFree(dev_data);

	return cudaStatus;
}
