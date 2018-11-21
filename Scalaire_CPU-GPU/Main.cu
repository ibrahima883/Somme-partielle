
#include "Matrix.h"
#include "PartialSum.h"


int main(int argc, char **argv)
{
	int nb_of_threads = atoi(argv[1]);
	omp_set_num_threads(nb_of_threads);
	float *V, *data;
	clock_t t0, t1;
	Matrix M(0, 0, nullptr);
	if (argc == 3)
	{
		const char * filename = argv[2];
		t0 = clock();
		M = Matrix(filename);
		t1 = clock();
		cout << "  Data loading time = " << (double)(t1 - t0) / CLOCKS_PER_SEC << " s" << endl;
	}
	else if (argc == 4)
	{
		t0 = clock();
		M.set_nbRows(atoi(argv[2]));
		M.set_nbColumns(atoi(argv[3]));
		M.set_data(atoi(argv[2]), atoi(argv[3]));
		t1 = clock();
		cout << "  Construction time of the matrix = " << (double)(t1 - t0) / CLOCKS_PER_SEC << " s" << endl;
	}
	else { cout << "  Incorrect number of arguments " << endl; }
	//cout << M << endl;

	int nb_rows = M.n;
	int nb_columns = M.m;
	int size = nb_rows*nb_columns;

	V = (float*)malloc(nb_columns * sizeof(float));
	data = (float*)malloc(size * sizeof(float));

	int i = 0;
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < nb_columns; ++i)
		V[i] = 0.;
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < size; ++i)
		data[i] = M.data[i];

	t0 = clock();
	cout << "  Allocation and Initialization time on the CPU = " << (double)(t0 - t1) / CLOCKS_PER_SEC << " s" << endl;

	if (nb_rows > 1000000) {
		int j;
#pragma omp parallel for schedule(static) private(j)
		for (j = 0; j < M.m; ++j) {
#pragma omp parallel for schedule(static) private(i)
			for (i = 1000000; i < M.n; ++i)
				V[j] += M.data[i*M.m + j];
		}
		nb_rows = 1000000;
	}

	// Add elements in parallel.
	cudaError_t cudaStatus = addWithCuda(V, data, nb_rows, nb_columns);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	t1 = clock();
	cout << "  GPU time of Partial Sum = " << (double)(t1 - t0) / CLOCKS_PER_SEC << " s" << endl;

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	ofstream f("PartialSum.csv");
	for (int j = 0; j < nb_columns; ++j)
	f << V[j] << endl;
	t0 = clock();
	cout << "  Backup time of the result vector = " << (double)(t0 - t1) / CLOCKS_PER_SEC << " s" << endl;
	
	free(data);
	free(V);

	system("pause>nul");
    return 0;
}
