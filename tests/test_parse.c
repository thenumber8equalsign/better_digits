#include <config.h>

#include <parsers.h>
#include <model.h>
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

/* make gradient (my kitty background is 0x212425) */
void make_gradient(char *buf, double val)
{
	uint8_t r1 = TERM_BACKGROUND >> 16 & 0xff;
	uint8_t g1 = TERM_BACKGROUND >> 8 & 0xff;
	uint8_t b1 = TERM_BACKGROUND & 0xff;

	uint8_t r2 = 0xff;
	uint8_t g2 = 0xff;
	uint8_t b2 = 0xff;

	unsigned long num_steps = 4096;
	size_t index;

	double r_slope = (double)(r2 - r1) / (num_steps - 1);
	double g_slope = (double)(g2 - g1) / (num_steps - 1);
	double b_slope = (double)(b2 - b1) / (num_steps - 1);

	uint8_t r[num_steps];
	uint8_t g[num_steps];
	uint8_t b[num_steps];

	memset(r, 0, sizeof(r));
	memset(g, 0, sizeof(g));
	memset(b, 0, sizeof(b));

	for (size_t i = 0; i < num_steps; ++i) {
		r[i] = r1 + r_slope * i;
		g[i] = g1 + g_slope * i;
		b[i] = b1 + b_slope * i;
	}

	index = val * (num_steps - 1);
	assert(index < num_steps);
	/* i don't give a damn if this is "bad practice" this is just the */
	/* test case */
	sprintf(buf, "\033[48;2;%d;%d;%dm", (int)r[index], (int)g[index],
		(int)b[index]);
}

int main(void)
{
	struct image *imgs;
	ssize_t ret;
	ssize_t len;
	size_t img_index = 10;

	ret = get_num_training_images();
	if (ret == -1)
		return EXIT_FAILURE;

	imgs = malloc(ret * sizeof(struct image));
	if (imgs == NULL)
		err(EXIT_FAILURE, "malloc");
	len = ret;

	printf("len: %zd\n", len);

	ret = parse_training_data(imgs, len);
	if (ret == -1)
		return EXIT_FAILURE; /* the OS is my garbage collector */

	printf("label: %c\n", imgs[img_index].label);

	for (size_t i = 0; i < IMAGE_HEIGHT; ++i) {
		for (size_t j = 0; j < IMAGE_WIDTH; ++j) {
			char buf[4096];
			make_gradient(buf, imgs[img_index].img[i][j]);
			printf("%s", buf);
			printf(" ");
			printf("\033[0m");
		}
		printf("\n");
	}

	return 0;
}
