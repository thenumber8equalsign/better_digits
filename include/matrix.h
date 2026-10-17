#ifndef MATRIX_H
#define MATRIX_H

#include <config.h>

#include <stddef.h>
#include <stdbool.h>

#define DIMENSION_ERROR 1

struct matrix {
	double *data;
	size_t rows;
	size_t cols;
};

/**
 * matrix_init() - initialize a matrix
 * @mat: the matix
 * @r: rows
 * @c: cols
 *
 * Return: 0 on success, nonzero on error
 */
int matrix_init(struct matrix *mat, size_t r, size_t c);

void matrix_free(struct matrix *mat);

int matrix_transpose(struct matrix *res, struct matrix *mat);

int matrix_multiply(struct matrix *result, const struct matrix *a,
		    const struct matrix *b);

int matrix_add(struct matrix *result, const struct matrix *a,
	       const struct matrix *b);

int matrix_in_place_add(struct matrix *a, const struct matrix *b);

static inline double matrix_get(const struct matrix *mat, size_t r, size_t c)
{
	return mat->data[mat->cols * r + c];
}

static inline void matrix_set(struct matrix *mat, size_t r, size_t c,
			      double val)
{
	mat->data[mat->cols * r + c] = val;
}

#endif
