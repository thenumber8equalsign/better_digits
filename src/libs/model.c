#include <config.h>

#include <assert.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <model.h>
#include <matrix.h>
#include <stddef.h>
#include <sys/types.h>
#include <stdlib.h>
#include <sys/random.h>
#include <limits.h>
#include <stdbool.h>

static inline double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

static inline double sigmoid_derivative(double x)
{
	return exp(-x) / (exp(-2 * x) + 2 * exp(-x) + 1);
}

static inline double sigmoid_inverse(double x)
{
	double (*ln)(double) = log;
	return ln(x) - ln(1 - x);
}

static inline double dot_product(const double *a, const double *b, size_t len)
{
	double res;

	res = 0;
	for (size_t i = 0; i < len; ++i)
		res += a[i] * b[i];

	return res;
}

/* floor a to a multiple of b */
static inline unsigned long floor_to_multiple(unsigned long a, unsigned long b)
{
	a /= b;
	a *= b;
	return a;
}

/* generate within [0, max) */
/* return error -1 on error and *val will be set to 0 */
/* *val will be set to the random value */
static inline int uniform_random(unsigned long *val, const unsigned long max)
{
	unsigned long b;
	unsigned long _max;
	ssize_t ret;
	_Bool run;

	_max = floor_to_multiple(ULONG_MAX, max);
	/* if num >= max, discard it */
	do {
		ret = getrandom(&b, sizeof(unsigned long), 0);
		if (ret != sizeof(unsigned long)) {
			*val = 0;
			return -1;
		}

		run = b >= _max;
	} while (run);
	*val = b % max;
	return 0;
}

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

void model_free(struct model *m)
{
	for (size_t i = 0; i < m->N; ++i) {
		layer_free(&m->layers[i]);
	}

	free(m->layers);
	m->layers = NULL;
	m->N = 0;
}

int model_params_init(struct model *m)
{
	/* LeCun initialization: sample within x \in [-\sqrt{\frac{3}{n_{l-1}}},
	 * 	+\sqrt{\frac{3}{n_{l-1}}}]
	 * for a uniform random distribution
	 */
	const double max = UINT32_MAX;
	double LeCun;
	double d_rand;
	unsigned long i_rand;
	ssize_t ret;
	int sign;
	size_t prev_n;

	for (size_t i = 1; i < m->N; ++i) {
		prev_n = m->layers[i - 1].N;
		LeCun = sqrt(3.0 / prev_n);

		/* biases */
		for (size_t j = 0; j < m->layers[i].N; ++j) {
			ret = uniform_random(&i_rand, 1UL << 33);
			if (ret != 0)
				return -1;
			sign = (i_rand & 1) ? -1 : 1;
			i_rand >>= 1;

			d_rand = (double)i_rand / max;
			d_rand *= sign;
			d_rand *= LeCun;

			m->layers[i].biases.data[j] = d_rand;
		}

		/* weights */
		for (size_t j = 0; j < m->layers[i].N * prev_n; ++j) {
			ret = uniform_random(&i_rand, 1UL << 33);
			if (ret != 0)
				return -1;
			sign = (i_rand & 1) ? -1 : 1;
			i_rand >>= 1;

			d_rand = (double)i_rand / max;
			d_rand *= sign;
			d_rand *= LeCun;
			m->layers[i].weights.data[j] = d_rand;
		}
	}
	return 0;
}

int model_cost(struct model *m, const struct image *img, const double *expctd,
	       const size_t len, double *cost)
{
	/* C = \frac{1}{2}\sum_j \left(y_j - a^L\right)^2 */
	ssize_t ret;

	*cost = 0;

	ret = model_compute(m, img);
	if (ret != 0)
		return -1;

	for (size_t i = 0; i < len; ++i) {
		double tmp;
		tmp = m->layers[m->N - 1].neurons.data[i];
		tmp -= expctd[i];
		tmp *= tmp;
		*cost += tmp;
	}

	*cost /= 2;

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

static inline int single_back_propagate(const struct image *img,
					struct train_params *params)
{
	/* \delta^L = \left(a^L - y\right) \odot \sigma'\left(z^L\right) */
	/* \delta^l = \left(\left(w^{l+1}\right)^T\delta^{l+1}\right) */
	/*	\odot \sigma'\left(z^l\right) */
	/* \frac{\partial C}{\partial b_j^l} = \delta_j^l */
	/*	the derivative with respect to bias is the error */
	/* \frac{\partial C}{\partial w_{jk}^l} = a_k^{l - 1} \delta_j^l */

	struct model *m = params->m;
	struct matrix *errors = params->errors;
	size_t num_layers = m->N;
	size_t index = 0;

	for (size_t j = 0; j < errors[num_layers - 1].rows; ++j) {
		double *ptr = &errors[num_layers - 1].data[j];
		double act = m->layers[num_layers - 1].neurons.data[j];
		double inv = sigmoid_inverse(act);
		double expc = params->expected[j];

		*ptr = (act - expc) * sigmoid_derivative(inv);
	}

	for (ssize_t i = num_layers - 2; i > 0; --i) {
		const double *neurons = m->layers[i].neurons.data;
		for (size_t j = 0; j < errors[i].rows; ++j) {
			const size_t cols = params->transposed_weights[i].cols;
			const double *data = params->transposed_weights[i].data;
			const double *weight_row = &data[j * cols];
			const double *next_ers = errors[i + 1].data;
			const double inv = sigmoid_inverse(neurons[j]);

			assert(cols == errors[i + 1].rows);

			const double dot =
				dot_product(weight_row, next_ers, cols);
			errors[i].data[j] = dot * sigmoid_derivative(inv);
		}
	}

	for (size_t i = 1; i < num_layers; ++i) {
		const double *prev_neurons = m->layers[i - 1].neurons.data;
		const size_t rows = m->layers[i].weights.rows;
		const size_t cols = m->layers[i].weights.cols;

		/* partial derivative for the weight */
		for (size_t flat = 0; flat < rows * cols; ++flat) {
			const size_t j = flat / cols;
			const size_t k = flat % cols;
			params->derivatives[index++] =
				prev_neurons[k] * errors[i].data[j];
		}

		/* partial derivative for the bias */
		for (size_t j = 0; j < errors[i].rows; ++j)
			params->derivatives[index++] = errors[i].data[j];
	}

	return 0;
}

int model_train(struct train_params *params)
{
	/* \delta^L = \left(a^L - y\right) \odot \sigma'\left(z^L\right) */
	/* \delta^l = \left(\left(w^{l+1}\right)^T\delta^{l+1}\right) */
	/*	\odot \sigma'\left(z^l\right) */
	/* \frac{\partial C}{\partial b_j^l} = \delta_j^l */
	/*	the derivative with respect to bias is the error */
	/* \frac{\partial C}{\partial w_{jk}^l} = a_k^{l - 1} \delta_j^l */

	ssize_t ret;
	struct model *m = params->m;
	for (size_t i = params->min; i < params->max; ++i) {
		const struct image *img =
			&params->training_data[params->indicies[i]];
		const char label = img->label;
		const size_t num_outputs = m->layers[m->N - 1].N;
		size_t out_index;

		assert(!char_to_output_neuron(&out_index, label));

		for (size_t j = 0; j < num_outputs; ++i) {
			params->expected[j] = 0;
			if (j == out_index)
				params->expected[j] = 1;
		}

		model_compute(params->m, img);

		ret = single_back_propagate(img, params);
		if (ret != 0)
			return -1;

		for (size_t j = 0; j < m->N; ++j) {
			struct matrix *d;
			d = &m->layers[j].neurons;
			double *mat = d->data;
			memset(mat, 0, d->rows * sizeof(double));
		}
	}
	return 0;
}
