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
#include <ctype.h>
#include <string.h>

enum operation {
	OP_ERR,
	OP_EXIT,
	OP_TRAIN,
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
	const char *opts[] = { "Train new model", "filler" };
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

		mvwprintw(w, i + 2, 0, "%-20s", opts[i]);
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
			mvwprintw(w, i + 2, 0, "%-20s", opts[i]);
		}

		wrefresh(w);
	}
outer:
	werase(w);
	wrefresh(w);

	wattroff(w, A_STANDOUT);
	switch (selected) {
	case 0:
		return OP_TRAIN;
		break;
	case 1:
		return OP_NOP;
		break;
	default:
		return OP_ERR;
		break;
	}
}

int main(int argc, char **argv)
{
	size_t training_len;
	size_t testing_len;
	ssize_t ret;
	int ch;

	enum operation op;

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

	op = prompt_user_for_op(w);

	switch (op) {
	case OP_TRAIN:
		mvwprintw(w, 0, 0, "You have chosen to train a new model");
		break;
	case OP_NOP:
		mvwprintw(w, 0, 0, "You have chosen my super secret option");
		break;
	case OP_ERR:
		mvwprintw(w, 0, 0, "i have no idea what's goin' on");
		break;
	case OP_TEST:
		mvwprintw(w, 0, 0, "You have chosen to test the model");
		break;
	case OP_EXIT:
	default:
		mvwprintw(w, 0, 0, "i have no idea what's goin' on");
		break;
	}

	/* temporary learning stuff */
	mvwprintw(w, 1, 0, "type \"quit\" and then press enter to exit");
	wmove(w, 2, 0);
	curs_set(1);
	echo();
	char buf[4096] = {0};
	size_t ind = 0;
	for (;;) {
		ch = wgetch(w);
		if (ch < 0 || ch == ERR || ch % 256 != ch)
			continue;

		if (ch == '\n') {
			wmove(w, 2, 0);
			wclrtoeol(w);
			if (strcmp(buf, "quit") == 0)
				break;
			memset(buf, 0, sizeof(buf));
			ind = 0;
		}

		if (!isprint(ch))
			continue;

		buf[ind++] = (char)ch;
	}

	delwin(w);
	endwin();
	free(training);
	free(testing);
	return 0;
}
