#include <config.h>

#include <matrix.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

int matrix_init(struct matrix *mat, size_t r, size_t c)
{
	mat->data = malloc(sizeof(double) * r * c);

	if (mat->data == NULL)
		return -1;

	memset(mat->data, 0, r * c * sizeof(double));
	mat->rows = r;
	mat->cols = c;
	return 0;
}

int matrix_transpose(struct matrix *res, struct matrix *mat)
{
	if (res->cols != mat->rows)
		return DIMENSION_ERROR;

	if (res->rows != mat->cols)
		return DIMENSION_ERROR;

	for (size_t i = 0; i < res->rows; ++i) {
		for (size_t j = 0; j < res->cols; ++j) {
			double m = matrix_get(mat, j, i);
			matrix_set(res, i, j, m);
		}
	}

	return 0;
}

void matrix_free(struct matrix *mat)
{
	if (mat->data != NULL)
		free(mat->data);
	mat->rows = 0;
	mat->cols = 0;
	mat->data = NULL;
}

int matrix_multiply(struct matrix *result, const struct matrix *a,
		    const struct matrix *b)
{
	if (a->cols != b->rows)
		return DIMENSION_ERROR;

	if (result->rows != a->rows || result->cols != b->cols)
		return DIMENSION_ERROR;

	// naievite is meh
	for (size_t i = 0; i < result->rows; ++i) {
		for (size_t j = 0; j < result->cols; ++j) {
			double accum = 0;
			for (size_t k = 0; k < a->cols; ++k) {
				const double _a = matrix_get(a, i, k);
				const double _b = matrix_get(b, k, j);
				accum += _a * _b;
			}
			matrix_set(result, i, j, accum);
		}
	}
	return 0;
}

int matrix_add(struct matrix *result, const struct matrix *a,
	       const struct matrix *b)
{
	if (a->cols != b->cols)
		return DIMENSION_ERROR;

	if (a->rows != b->rows)
		return DIMENSION_ERROR;

	if (result->rows != a->rows)
		return DIMENSION_ERROR;

	if (result->cols != a->cols)
		return DIMENSION_ERROR;

	for (size_t i = 0; i < a->rows; ++i) {
		for (size_t j = 0; j < a->cols; ++j) {
			const double _a = matrix_get(a, i, j);
			const double _b = matrix_get(b, i, j);
			matrix_set(result, i, j, _a + _b);
		}
	}
	return 0;
}

int matrix_in_place_add(struct matrix *a, const struct matrix *b)
{
	if (a->cols != b->cols)
		return DIMENSION_ERROR;

	if (a->rows != b->rows)
		return DIMENSION_ERROR;

	for (size_t i = 0; i < a->rows; ++i) {
		for (size_t j = 0; j < a->cols; ++j) {
			const double _a = matrix_get(a, i, j);
			const double _b = matrix_get(b, i, j);
			matrix_set(a, i, j, _a + _b);
		}
	}
	return 0;
}
