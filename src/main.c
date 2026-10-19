#include <config.h>

#include <limits.h>
#include <model.h>
#include <matrix.h>
#include <parsers.h>
#include <stdlib.h>
#include <err.h>
#include <stdio.h>
#include <assert.h>
#include <ncurses.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>
#include <locale.h>

enum operation {
	OP_ERR,
	OP_EXIT,
	OP_TRAINNEW,
	OP_CONTTRAIN,
	OP_NOP,
	OP_TEST,
	/* there will be more operations, like load params from file & test */
	/* or continue training, and etc. */
};

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

enum operation prompt_user_for_op(WINDOW *w)
{
	const char *opts[] = {
		"Train new model",
		"Continue training existing model",
	};
	size_t opts_len = sizeof(opts) / sizeof(char *);
	ssize_t selected = 0;
	enum operation op;
	int ch;

	wattroff(w, A_STANDOUT);
	mvwprintw(w, 0, 0, "Please select an option");

	for (size_t i = 0; i < opts_len; ++i) {
		if (selected == i)
			wattron(w, A_STANDOUT);
		else
			wattroff(w, A_STANDOUT);

		mvwprintw(w, i + 2, 0, "%-40s", opts[i]);
	}

	wrefresh(w);

	for (;;) {
		ch = wgetch(w);

		switch (ch) {
		case 'q':
			wattroff(w, A_STANDOUT);
			werase(w);
			return OP_EXIT;
			break;
		case KEY_DOWN:
		case 'j':
			--selected;
			break;
		case 'k':
		case KEY_UP:
			++selected;
			break;
		case '\n':
			goto outer;
		default:
			continue;
		}

		if (selected >= opts_len) {
			selected %= opts_len;
		} else if (selected < 0) {
			selected *= -1;
			selected %= opts_len;
			selected = opts_len - selected;
		}

		werase(w);

		wattroff(w, A_STANDOUT);
		mvwprintw(w, 0, 0, "Please select an option");

		for (size_t i = 0; i < opts_len; ++i) {
			if (i == selected)
				wattron(w, A_STANDOUT);
			else
				wattroff(w, A_STANDOUT);
			mvwprintw(w, i + 2, 0, "%-40s", opts[i]);
		}

		wrefresh(w);
	}
outer:
	werase(w);
	wrefresh(w);

	wattroff(w, A_STANDOUT);
	switch (selected) {
	case 0:
		return OP_TRAINNEW;
		break;
	case 1:
		return OP_CONTTRAIN;
		break;
	default:
		return OP_ERR;
		break;
	}
}

void train_new_model(WINDOW *w, struct image *train, struct image *test,
		     size_t train_len, size_t test_len)
{
	/* ask for number of layers */
	ssize_t ret;
	int ch;
	char buf[4096]; /* TODO: re-add my old array code */
	ssize_t ind;

	ind = 0;
	memset(buf, 0, sizeof(buf));

	mvwprintw(w, 0, 0, "How many hidden layers should this model have?");
	wmove(w, 1, 0);
	echo();
	curs_set(1);
	nodelay(w, FALSE);
	wrefresh(w);

	for (;;) {
		ch = wgetch(w);
		if (ch == KEY_BACKSPACE) {
			--ind;
			ind = (ind < 0) ? 0 : ind;
			buf[ind] = 0;
			wdelch(w);
		}
		if (ch == '\n')
			break;
		if (ch % UCHAR_MAX != ch || !isprint(ch))
			continue;
		buf[ind++] = (char)ch;
		wrefresh(w);
	}

	werase(w);
	wrefresh(w);
	noecho();
	curs_set(0);
	nodelay(w, TRUE);

	mvwprintw(w, 0, 0, "%s", buf);
}

int main(int argc, char **argv)
{
	size_t train_len;
	size_t test_len;
	ssize_t ret;
	int ch;
	_Bool valid_input;

	enum operation op;

	struct image *training;
	struct image *testing;

	WINDOW *w;

	printf("Please wait while training and testing data is parsed\n");
	ret = prepare_data(&training, &testing, &train_len, &test_len);
	if (ret != 0)
		return ret;

	setlocale(LC_ALL, "");

	initscr();
	w = newwin(0, 0, 0, 0);

	noecho();
	keypad(w, TRUE);
	curs_set(0);
	nodelay(w, TRUE);

	op = prompt_user_for_op(w);

	switch (op) {
	case OP_TRAINNEW:
		train_new_model(w, training, testing, train_len, test_len);
		break;
	case OP_CONTTRAIN:
		break;
	case OP_NOP:
		break;
	case OP_TEST:
		break;
	case OP_EXIT:
		goto cleanup;
		break;
	case OP_ERR:
	default:
		mvwprintw(w, 0, 0, "i have no idea what's goin' on");
		goto prompt_for_exit;
		break;
	}

prompt_for_exit:
	mvwprintw(w, 1, 0, "press q to exit");
	wmove(w, 2, 0);
	wrefresh(w);
	for (;;) {
		ch = wgetch(w);
		if (ch == 'q')
			break;
	}

cleanup:
	delwin(w);
	endwin();
	free(training);
	free(testing);
	training = NULL;
	testing = NULL;

	return 0;
}
