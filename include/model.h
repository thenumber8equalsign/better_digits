#ifndef MODEL_H
#define MODEL_H
#include <config.h>
#include <matrix.h>
#include <stddef.h>
#include <math.h>

#define IMAGE_WIDTH (28)
#define IMAGE_HEIGHT (28)

static inline double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

static inline void vectorized_sigmoid(double *out, double *v, size_t len)
{
	for (size_t i = 0; i < len; ++i) {
		out[i] = 1.0 / (1.0 + exp(-v[i]));
	}
}

static inline double sigmoid_derivative(double x)
{
	return exp(-x) / (exp(-2 * x) + 2 * exp(-x) + 1);
}

static inline void vectorized_sigmoid_derivative(double *out, double *v,
						 size_t len)
{
	for (size_t i = 0; i < len; ++i) {
		out[i] = sigmoid_derivative(v[i]);
	}
}

static inline double sigmoid_inverse(double x)
{
	double (*ln)(double) = log;
	return ln(x) - ln(1 - x);
}

/**
 * struct layer - neural network layer
 * @weights: the weights
 * @biases: the biases
 * @neurons: the neuron activations
 * @N: the number of neurons in this layer
 */
struct layer {
	struct matrix weights; /* N * (prev. N) matrix */
	struct matrix biases; /* N * 1 matrix */
	struct matrix neurons; /* N * 1 matrix */
	size_t N; /* number of neurons in this layer */
};

/**
 * struct model - the model (it's just an array of layers)
 * @layers: the array of layers
 * @N: the number of layers
 */
struct model {
	struct layer *layers; /* array of layers */
	size_t N; /* length of array layers */
};

/**
 * struct image - an image representing a digit
 * @label: the integer label, ascii code for the character
 * @img: 2D array containing the grayscale values for the image
 */
struct image {
	char label;
	double img[IMAGE_HEIGHT][IMAGE_WIDTH];
};

/**
 * struct train_params - parameters to the model_train function
 * @m: the model
 * @training_data: the training data
 * @derivatives: output for all derivates, format is as followed:
 * 	each layer:
 * 		flattened matrix weights, row major order
 * 		biases
 * 	... repeat for each layer
 * 	must be long enough, length = total number weights and biases
 * @errors: for use by the function, allows not calling malloc() in the function
 * 	length must be at least total number of neurons
 *	the data in this must be set to 0
 * @num_neurons: the total number of neurons in the entire model
 * 	(also the length of the @errors array)
 * @expected: for use by the function, should have length of number of neurons
 * 	in last layer
 * @transposed_weights: array of all the transpose matricies of the weights
 * @indicies: array of all indicies into the @training_data array,
 * 	should be same length as @training_data, used for shuffling reasons
 * 	(much easier to shuffle array of integers than it is an array of
 * 	structs)
 * @min: the minimum index into the array @indicies to start at
 * @max: the (exclusive) maximum into the array @indicies
 */
struct train_params {
	struct model *m;
	const struct image *training_data;
	double *derivatives;
	double *errors;
	size_t num_neurons;
	double *expected;
	const struct matrix *transposed_weights;
	const size_t *indicies;
	const size_t min;
	const size_t max;
};

/**
 * char_to_output_neuron() - given a character, return the output neuron index
 * @ch: the character
 * @result: the result
 *
 * this function gets the index of the last layer neuron that corrosponds to the
 * char ch, and puts it in *result
 *
 * Return: -1 if ch is invalid, 0 for success
 */
static inline int char_to_output_neuron(size_t *result, const char ch)
{
	if (ch < '0' || ch > '9')
		return -1;

	*result = ch - 0x30;
	return 0;
}

/**
 * output_neuron_to_char() - the inverse of the above function
 * @ch: the result
 * @neur: the index of the neuron
 *
 * See the above's function documentation
 *
 * Return: -1 if neur is invalid, 0 for success
 */
static inline int output_neuron_to_char(char *ch, const size_t neur)
{
	if (neur > 9)
		return -1;
	*ch = neur + 0x30;
	return 0;
}

/**
 * layer_init() - initializer a layer
 * @l: the layer to initialize
 * @N the number of neurons in this layer
 * @prev_n: the number of neurons in the previous layer, or 0
 *
 * this function initializes a layer's matricies and its values
 * everything is initalized to zero
 *
 * if @prev_n is 0, weights and biases will not be initialized, and set to NULL
 *
 * Return: 0 for success, nonzero on error
 */
int layer_init(struct layer *l, size_t n, size_t prev_n);

/**
 * layer_free() - this does exactly what you think it does
 * @l: the layer
 */
void layer_free(struct layer *l);

/**
 * layer_compute() - compute the neurons of a layer
 * @cur: the layer to compute
 * @prev: the previous layer
 *
 * This function uses \sigma\left(x\right) = \frac{1}{1+e^{-x}}
 *
 * Return: 0 for success, nonzero on error
 */
int layer_compute(struct layer *cur, const struct layer *prev);

/**
 * model_init() - initialize a model
 * @m: the model
 * @num_neurons: the array of number of neurons in each layer
 * @n: the number of layers (also the length of the @num_neurons array)
 *
 * Return: 0 on success, nonzero on error
 */
int model_init(struct model *m, size_t *num_neurons, size_t n);

/**
 * model_params_init() - initialize the parameters
 * @m: the model
 *
 * Return: 0 on success, nonzero on error
 */
int model_params_init(struct model *m);

/**
 * model_cost() - compute the cost for a given image
 * @m: the model
 * @img: the image
 * @expctd: expected output
 * @len: length of @expctd array; should be same as number of neurons in last
 * 	layer
 * @cost: will be set to the cost
 *
 * cost is the sum of the squares of the differences of expected output
 * and received output
 *
 * Return: 0 on success, -1 on failure
 */
int model_cost(struct model *m, const struct image *img, const double *expctd,
	       const size_t len, double *cost);

/**
 * model_compute() - compute all the layers
 * @m: the model to compute
 * @img: the image to use as input
 *
 * Return: 0 for success, nonzero on error
 */
int model_compute(struct model *m, const struct image *img);

/**
 * model_train() - use backpropagation
 * @params: parameters (see above definition of the struct)
 *
 * Return: 0 on success, nonzero on error
 */
int model_train(struct train_params *params);

#endif
