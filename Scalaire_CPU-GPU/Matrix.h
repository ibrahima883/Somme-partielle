// +--------------------------------------------------------------------------+
// | File      : Matrix.h                                                     |
// | Utility   : declaration of the Matrix.                                   |
// | Author    : Ibrahima DIALLO                                              |
// | Creation  : 02.03.2017                                                   |                                                |
// +--------------------------------------------------------------------------+

#ifndef H_MATRIX_H
#define H_MATRIX_H

#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <fstream>
#include <ctime>
#include <cmath>

#define PI 3.14159

using namespace std;

class Matrix
{
public:
	int n, m;
	float *data;
	Matrix(int r, int c, float *d) : n(r), m(c), data(d) {};
	Matrix(const char *filename);
	void set_nbRows(int r) { n = r; }
	void set_nbColumns(int c) { m = c; }
	void set_data(int n, int m);
	Matrix(const Matrix &M);
	~Matrix();
	Matrix & operator=(Matrix const & M);
	float& operator[](int i) { return data[i]; };
	float operator[](int i) const { return data[i]; };
	friend ostream & operator<<(ostream & out, const Matrix & M);
};


#endif

