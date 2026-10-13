#include <config.h>
#include <model.h>
#include <matrix.h>
#include <stddef.h>
#include <sys/types.h>
#include <stdlib.h>

int layer_init(struct layer *l, size_t n, size_t prev_n)
{
	ssize_t ret;

	if (prev_n == 0)
		goto skip;

	ret = matrix_init(&l->weights, n, prev_n);
	if (ret == -1)
		return -1;

	ret = matrix_init(&l->biases, n, 1);
	if (ret == -1) {
		matrix_free(&l->weights);
		return -1;
	}
skip:
	ret = matrix_init(&l->neurons, n, 1);
	if (ret == -1) {
		matrix_free(&l->weights);
		matrix_free(&l->biases);
		return -1;
	}

	l->N = n;
	return 0;
}

void layer_free(struct layer *l)
{
	matrix_free(&l->weights);
	matrix_free(&l->biases);
	matrix_free(&l->neurons);
	l->N = 0;
}

int layer_compute(struct layer *cur, const struct layer *prev)
{
	ssize_t ret;

	ret = matrix_multiply(&cur->neurons, &cur->weights, &prev->neurons);
	if (ret != 0) {
		return -1;
	}

	ret = matrix_in_place_add(&cur->neurons, &cur->biases);
	if (ret != 0) {
		return -1;
	}

	for (size_t i = 0; i < cur->neurons.rows; ++i) {
		const double n = matrix_get(&cur->neurons, i, 0);
		matrix_set(&cur->neurons, i, 0, sigmoid(n));
	}
	return 0;
}

int model_init(struct model *m, size_t *num_neurons, size_t n)
{
	ssize_t ret;

	m->layers = malloc(n * sizeof(struct layer));
	if (m->layers == NULL)
		return -1;

	m->N = n;

	for (size_t i = 0; i < n; ++i) {
		if (i == 0)
			ret = layer_init(&m->layers[i], num_neurons[i], 0);
		else
			ret = layer_init(&m->layers[i], num_neurons[i],
					 num_neurons[i - 1]);

		if (ret != 0) {
			for (size_t j = 0; j < i; ++j)
				layer_free(&m->layers[j]);
			goto error;
		}
	}

	return 0;
error:
	free(m->layers);
	m->layers = NULL;
	m->N = 0;
	return -1;
}

int model_params_init(struct model *m)
{
	/* LeCun initialization: sample within x \in [-\sqrt{\frac{3}{n_{l-1}}},
	 * 	+\sqrt{\frac{3}{n_{l-1}}}]
	 * for a uniform random distribution
	 */
	return 0;
}

int model_compute(struct model *m, const struct image *img)
{
	ssize_t ret;

	for (size_t i = 0; i < IMAGE_HEIGHT; ++i) {
		for (size_t j = 0; j < IMAGE_WIDTH; ++j) {
			const size_t e = i * IMAGE_WIDTH + j;
			matrix_set(&m->layers[0].neurons, e, 0, img->img[i][j]);
		}
	}

	for (size_t i = 1; i < m->N; ++i) {
		ret = layer_compute(&m->layers[i], &m->layers[i - 1]);
		if (ret != 0)
			return -1;
	}

	return 0;
}

int model_train(struct model *m, struct image *training_data, size_t len);
