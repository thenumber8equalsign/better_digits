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
 * @label: the integer label, label \in [0, 9], not an ascii value for the digit
 * @img: 2D array containing the grayscale values for the image
 */
struct image {
	char label;
	double img[IMAGE_HEIGHT][IMAGE_WIDTH];
};

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
 * Return: 0 for success, -1 on error
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
 * Return: 0 for success, -1 on error
 */
int layer_compute(struct layer *cur, const struct layer *prev);

/**
 * model_init() - initialize a model
 * @m: the model
 * @num_neurons: the array of number of neurons in each layer
 * @n: the number of layers (also the length of the @num_neurons array)
 *
 * Return: 0 on success, -1 on failure
 */
int model_init(struct model *m, size_t *num_neurons, size_t n);

/**
 * model_compute() - compute all the layers
 * @m: the model to compute
 * @img: the image to use as input
 *
 * Return: 0 for success, -1 on error
 */
int model_compute(struct model *m, const struct image *img);

/**
 * model_train() - use backpropagation
 * @m: model to train
 * @training_data: training data to use
 * @len: length of @training_data
 *
 * Return: 0 on success, -1 on error
 */
int model_train(struct model *m, struct image *training_data, size_t len);
