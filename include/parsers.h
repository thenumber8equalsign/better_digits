#ifndef PARSERS_H
#define PARSERS_H

#include <config.h>

#include <model.h>
#include <stddef.h>
#include <sys/types.h>

#define DATA_PATH "./data/"
#define TRAINING_DATA_PATH (DATA_PATH "training_data/")
#define TESTING_DATA_PATH (DATA_PATH "testing_data/")

#define PARSE_ERROR (2)

ssize_t get_num_training_images();

ssize_t get_num_testing_images();

/* returns number of images parsed, or negative status code
 * -1 on library error */
ssize_t parse_testing_data(struct image *data, size_t dlen);

/* returns number of images parsed, or negative status code
 * -1 on library error */
ssize_t parse_training_data(struct image *data, size_t dlen);

#endif
