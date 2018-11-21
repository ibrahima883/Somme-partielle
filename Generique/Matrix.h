// +--------------------------------------------------------------------------+
// | File      : Matrix.h                                                     |
// | Utility   : declaration of the Matrix.                                   |
// | Author    : Ibrahima DIALLO                                              |
// | Creation  : 02.03.2017                                                   |                                                |
// +--------------------------------------------------------------------------+

#ifndef H_MATRIX_H
#define H_MATRIX_H

#ifdef _OPENMP
  #include <omp.h>
#endif
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <ctime>
#include <cmath>

#define PI 3.14159

using namespace std;

typedef float Y;

template <typename T>
class Matrix
{
public:
	int n, m;
	T *data;
	Matrix(int r, int c, float *d) : n(r), m(c), data(d) {};
	Matrix(const char *filename);
	void set_nbRows(int r) { n = r; }
	void set_nbColumns(int c) { m = c; }
	void set_data(int n, int m);
	Matrix(const Matrix<T> & M);
	~Matrix();
	Matrix<T> & operator=(const Matrix<T> & M);
	T & operator[](int i) { return data[i]; };
	T operator[](int i) const { return data[i]; };
	friend ostream & operator<<(ostream & out, const Matrix<T> & M){
		out << " Matrix " << M.n << "x" << M.m << endl;
		for (int i = 0; i < M.n*M.m; i++) {
			if (i >= M.m && i%M.m == 0) { out << endl; }
			out << " " << M.data[i] << " ";
		}
		return out;
	};
};

template <typename T>
Matrix<T>::Matrix(const char * filename)
{
	ifstream f(filename);
	assert(f);
	f >> n >> m;
	assert(f.good());
	int size = n*m, i = 0;
	data = new T[size];
	for (i = 0; i<size; ++i) {
		f >> data[i];
		assert(f.good());
	}
}

template <typename T>
void Matrix<T>::set_data(int n, int m) {
	int i = 0, size = n*m;
	data = new T[size];
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < size; i++)
		data[i] = (T)(cos(0.1*PI*i) + sin(0.1*PI*i));
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> & MatrixCopy) {
	n = MatrixCopy.n;
	m = MatrixCopy.m;
	int size = n*m, i = 0;
	data = new T[size];
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i< size; i++) {
		data[i] = MatrixCopy.data[i];
	}
}

template <typename T>
Matrix<T>::~Matrix() {
	delete[] data;
}

template<typename T>
Matrix<T> & Matrix<T>::operator=(const Matrix<T> & M) {
	if (&M == this) return *this;
	n = M.n;
	m = M.m;
	delete[] data;
	data = new T[n*m];
	int i = 0;
#pragma omp parallel for schedule(runtime) private(i)
	for (i = 0; i < n*m; ++i)
		data[i] = M.data[i];
	return *this;
}

#endif
