// +--------------------------------------------------------------------------+
// | File      : Operations.h                                                 |
// | Utility   : definition of operations.                                    |
// | Author    : Ibrahima DIALLO                                              |
// | Creation  : 04.03.2017                                                   |                                                |
// +--------------------------------------------------------------------------+

#ifndef H_OPERATIONS_H
#define H_OPERATIONS_H

template <typename T>
struct Addition
{
	static void Computation(T& Resultat, const T& Valeur)
	{
		Resultat += Valeur;
	}
};

template <typename T>
struct Multiplication
{
	static void Computation(T& Resultat, const T& Valeur)
	{
		Resultat *= Valeur;
	}
};

template <typename T>
struct Soustraction
{
	static void Computation(T& Resultat, const T& Valeur)
	{
		Resultat -= Valeur;
	}
};

template <typename T>
struct Division
{
	static void Computation(T& Resultat, const T& Valeur)
	{
		Resultat /= Valeur; //Be careful not to divide by zero
	}
};
template <typename T, typename Operation = Addition<T> >
struct Operations
{
	static T Add(const T* Debut, const T* Fin, int step)
	{
		T Resultat = 0;
		for (; Debut < Fin; Debut += step)
			Operation::Computation(Resultat, *Debut);

		return Resultat;
	}
	static T Multiply(const T* Debut, const T* Fin, int step)
	{
		T Resultat = 1;
		for (; Debut < Fin; Debut += step)
			Operation::Computation(Resultat, *Debut);

		return Resultat;
	}
	static T Subtract(const T* Debut, const T* Fin, int step)
	{
		T Resultat = 2 * (*Debut);
		for (; Debut < Fin; Debut += step)
			Operation::Computation(Resultat, *Debut);

		return Resultat;
	}
	static T Divide(const T* Debut, const T* Fin, int step)
	{
		T Resultat = (*Debut) * (*Debut);
		for (; Debut < Fin; Debut += step)
			Operation::Computation(Resultat, *Debut);

		return Resultat;
	}
};

#endif
