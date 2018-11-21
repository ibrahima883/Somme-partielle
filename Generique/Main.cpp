#include "Matrix.h"
#include "Operations.h"


int main(int argc, char **argv)
{
	int nb_of_threads = atoi(argv[1]);
	#ifdef _OPENMP
	  omp_set_num_threads(nb_of_threads);
	#else
	  nb_of_threads *= 0;
	#endif
	int j = 0;
	clock_t t0, t1 = clock();
	Matrix<Y> M(0, 0, nullptr);
	if (argc == 3)
	{
		const char * filename = argv[2];
		t0 = clock();
		M = Matrix<Y>(filename);
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
	Y * V = new Y[M.m], *begin = 0, *end = 0;
#pragma omp parallel for schedule(runtime) private(j,begin,end)
	for (j = 0; j < M.m; ++j) {
		begin = M.data + j;
		end = M.data + j + M.n*M.m;
		V[j] = Operations<Y, Addition<Y>>::Add(begin, end, M.m);
	}
	t0 = clock();
	cout << "  CPU time of Partial Sum = " << (double)(t0 - t1) / CLOCKS_PER_SEC << " s" << endl;

	// Saving the result to an excel file
	ofstream f("PartialSum.csv");
	for (int j = 0; j < M.m; ++j)
		f << V[j] << endl;
	
	t1 = clock();
	cout << "  Backup time of the result vector = " << (double)(t1 - t0) / CLOCKS_PER_SEC << " s" << endl;

	delete[] V;
	cout << endl << "Press any key to exit...";
	getchar();
	return 0;
}
