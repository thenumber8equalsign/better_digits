#include <config.h>

#include <model.h>
#include <matrix.h>
#include <parsers.h>
#include <stdlib.h>
#include <err.h>
#include <stdio.h>
#include <assert.h>

#include <ncurses.h>
#include <locale.h>

int prepare_data(struct image **training, struct image **testing,
		 size_t *training_len, size_t *testing_len)
{
	ssize_t ret;

	ret = get_num_training_images();
	if (ret == -1)
		return EXIT_FAILURE;
	*training_len = ret;

	ret = get_num_testing_images();
	if (ret == -1)
		return EXIT_FAILURE;
	*testing_len = ret;

	*training = malloc(*training_len * sizeof(struct image));
	if (training == NULL)
		err(EXIT_FAILURE, "malloc");

	*testing = malloc(*testing_len * sizeof(struct image));
	if (testing == NULL) {
		free(training);
		err(EXIT_FAILURE, "malloc");
	}

	ret = parse_training_data(*training, *training_len);
	if (ret < 0) {
		fprintf(stderr, "Error when parsing training data\n");
		free(training);
		free(testing);
		return EXIT_FAILURE;
	}
	assert(ret <= *training_len);
	*training_len = ret;

	ret = parse_testing_data(*testing, *testing_len);
	if (ret < 0) {
		fprintf(stderr, "Error when parsing testing data\n");
		free(training);
		free(testing);
		return EXIT_FAILURE;
	}
	assert(ret <= *testing_len);
	*testing_len = ret;
	return 0;
}

int main(int argc, char **argv)
{
	size_t training_len;
	size_t testing_len;
	ssize_t ret;
	int ch;

	struct image *training;
	struct image *testing;

	WINDOW *w;

	ret = prepare_data(&training, &testing, &training_len, &testing_len);
	if (ret != 0)
		return ret;

	setlocale(LC_ALL, "");

	initscr();
	w = newwin(0, 0, 0, 0);

	noecho();
	keypad(w, TRUE);
	curs_set(0);
	nodelay(w, TRUE);

	for (;;) {
		ch = wgetch(w);
		if (ch == 'q')
			break;
	}

	delwin(w);
	endwin();
	free(training);
	free(testing);
	return 0;
}
