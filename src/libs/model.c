#include <config.h>
#include <model.h>
#include <matrix.h>
#include <stddef.h>
#include <sys/types.h>

int layer_init(struct layer *l, size_t n, size_t prev_n)
{
	ssize_t ret;

	ret = matrix_init(&l->weights, n, prev_n);
	if (ret == -1)
		return -1;

	ret = matrix_init(&l->biases, n, 1);
	if (ret == -1) {
		matrix_free(&l->weights);
		return -1;
	}

	ret = matrix_init(&l->neurons, n, 1);
	if (ret == -1) {
		matrix_free(&l->weights);
		matrix_free(&l->biases);
		return -1;
	}

	l->N = n;
	return 0;
}
