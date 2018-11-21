// +--------------------------------------------------------------------------+
// | File      : PartialSum.h                                                 |
// | Utility   : declarations of functions.                                   |
// | Author    : Ibrahima DIALLO                                              |
// | Creation  : 06.03.2017                                                   |                                                |
// +--------------------------------------------------------------------------+

#ifndef H_PARTIALSUM_H
#define H_PARTIALSUM_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <iostream>

#define SIZE_BLOCK_X 1024

using namespace std;

cudaError_t addWithCuda(float *V, float *data, int nb_rows, int nb_columns);

__global__ void PartialSum_Kernel(float *V, float *data, int nb_rows, int nb_columns);

#endif