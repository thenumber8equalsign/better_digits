#include <config.h>

#include <matrix.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

void print_mat(struct matrix *mat);

int main(void)
{
	struct matrix a;
	struct matrix b;
	struct matrix c;
	memset(&a, 0, sizeof(a));
	memset(&b, 0, sizeof(b));
	memset(&c, 0, sizeof(c));

	assert(matrix_init(&a, 4, 3) == 0);
	assert(matrix_init(&b, 3, 3) == 0);
	assert(matrix_init(&c, 4, 3) == 0);

	matrix_set(&a, 0, 0, 1);
	matrix_set(&a, 0, 1, 0);
	matrix_set(&a, 0, 2, 1);
	matrix_set(&a, 1, 0, 2);
	matrix_set(&a, 1, 1, 1);
	matrix_set(&a, 1, 2, 1);
	matrix_set(&a, 2, 0, 0);
	matrix_set(&a, 2, 1, 1);
	matrix_set(&a, 2, 2, 1);
	matrix_set(&a, 3, 0, 1);
	matrix_set(&a, 3, 1, 1);
	matrix_set(&a, 3, 2, 2);

	matrix_set(&b, 0, 0, 1);
	matrix_set(&b, 0, 1, 2);
	matrix_set(&b, 0, 2, 1);
	matrix_set(&b, 1, 0, 2);
	matrix_set(&b, 1, 1, 3);
	matrix_set(&b, 1, 2, 1);
	matrix_set(&b, 2, 0, 4);
	matrix_set(&b, 2, 1, 2);
	matrix_set(&b, 2, 2, 2);

	assert(matrix_multiply(&c, &a, &b) == 0);

	print_mat(&c);

	matrix_free(&a);
	matrix_free(&b);
	matrix_free(&c);

	return 0;
}

void print_mat(struct matrix *mat)
{
	for (size_t i = 0; i < mat->rows; ++i) {
		for (size_t j = 0; j < mat->cols; ++j) {
			printf("%f ", matrix_get(mat, i, j));
		}
		printf("\n");
	}
}
