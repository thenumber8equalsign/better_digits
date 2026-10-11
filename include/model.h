#include <config.h>
#include <matrix.h>
#include <stddef.h>

struct layer {
	struct matrix weights; /* N * (prev. N) matrix */
	struct matrix biases; /* N * 1 matrix */
	struct matrix neurons; /* N * 1 matrix */
	size_t N; /* number of neurons in this layer */
};

struct model {
	struct layer *layers; /* array of layers */
	size_t N; /* length of array layers */
};

/**
 * layer_init() - initializer a layer
 * @l: the layer to initialize
 * @N the number of neurons in this layer
 * @prev_N: the number of neurons in the previous layer
 *
 * this function initializes a layer's matricies and its values
 * everything is initalized to zero
 *
 * Return: 0 for success, -1 on error
 */
int layer_init(struct layer *l, size_t n, size_t prev_n);
