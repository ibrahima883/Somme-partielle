// +--------------------------------------------------------------------------+
// | File      : Matrix.cpp                                                   |
// | Utility   : definition of the Matrix.                                    |
// | Author    : Ibrahima DIALLO                                              |
// | Creation  : 02.03.2017                                                   |                                                |
// +--------------------------------------------------------------------------+

#include "Matrix.h"


Matrix::Matrix(const char * filename)
{
    ifstream f(filename); 
    assert( f);
    f >> n >> m;
    assert( f.good());
	int size = n*m, i = 0;
    data = new float[size];
    for(i=0; i<size;++i){
		f >> data[i];
		assert(f.good());
    }
}

Matrix::Matrix(const Matrix & MatrixCopy) {
    n = MatrixCopy.n;
    m = MatrixCopy.m;
	int size = n*m, i = 0;
    data = new float[size]; 
#pragma omp parallel for schedule(runtime) private(i)
    for(i = 0; i< size; i++) {
            data[i] = MatrixCopy.data[i];
       }
}

void Matrix::set_data(int n, int m) {
	int i = 0, size = n*m;
	data = new float[size];
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < size; i++)
		data[i] = (float)(cos(0.1*PI*i) + sin(0.1*PI*i));
}

Matrix::~Matrix() {
	delete[] data;
}

ostream & operator<<(ostream & out, const Matrix & M){
  out << " Matrix " << M.n << "x" << M.m << endl;
  for(int i = 0; i < M.n*M.m; i++){
    if (i >= M.m && i%M.m ==0) { out << endl;}
    out << " " << M.data[i] << " ";
  }
  return out;
}

Matrix & Matrix::operator=(Matrix const & M) {
	if (&M == this) return *this;
	n = M.n;
	m = M.m;
	delete[] data;
	data = new float[n*m];
	int i = 0;
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < n*m; ++i)
		data[i] = M.data[i];
	return *this;
}
