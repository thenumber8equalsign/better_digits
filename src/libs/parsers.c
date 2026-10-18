#include <config.h>

#include <parsers.h>
#include <model.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <err.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <stdbool.h>

ssize_t get_num_training_images()
{
	ssize_t num;
	_Bool run;
	DIR *dir = NULL;
	struct dirent *ent = NULL;

	num = 0;
	dir = opendir(TRAINING_DATA_PATH);
	if (dir == NULL) {
		warn("opendir");
		return -1;
	}

	do {
		errno = 0;
		ent = readdir(dir);
		if (ent == NULL && errno == 0) {
			run = false;
			break;
		} else if (ent == NULL) {
			warn("readdir");
			return -1;
		}

		if (ent->d_type != DT_REG)
			continue;

		run = true;
		++num;
	} while (run);

	closedir(dir);
	dir = NULL;
	ent = NULL;

	return num;
}

ssize_t get_num_testing_images()
{
	ssize_t num;
	_Bool run;
	DIR *dir = NULL;
	struct dirent *ent = NULL;

	num = 0;
	dir = opendir(TESTING_DATA_PATH);
	if (dir == NULL) {
		warn("opendir");
		return -1;
	}

	do {
		errno = 0;
		ent = readdir(dir);
		if (ent == NULL && errno == 0) {
			run = false;
			break;
		} else if (ent == NULL) {
			warn("readdir");
			return -1;
		}

		if (ent->d_type != DT_REG)
			continue;

		run = true;
		++num;
	} while (run);

	closedir(dir);
	dir = NULL;
	ent = NULL;

	return num;
}

static inline ssize_t parse_image(double img[static IMAGE_HEIGHT][IMAGE_WIDTH],
				  char path[static PATH_MAX])
{
	ssize_t ret;
	char label;
	char raw_img[IMAGE_HEIGHT][IMAGE_WIDTH];
	int fd;

	fd = open(path, O_RDONLY);
	if (fd == -1) {
		warn("open");
		return -1;
	}

	ret = read(fd, &label, 1);
	if (ret == -1) {
		warn("read");
		close(fd);
		return -1;
	}

	if (char_to_output_neuron(NULL, label) != 0) {
		close(fd);
		return -PARSE_ERROR;
	}

	ret = lseek(fd, 24, SEEK_SET);
	if (ret == -1) {
		warn("lseek");
		close(fd);
		return -1;
	}

	ret = read(fd, raw_img, IMAGE_HEIGHT * IMAGE_WIDTH);
	if (ret == -1) {
		warn("lseek");
		close(fd);
		return -1;
	}

	for (size_t i = 0; i < IMAGE_HEIGHT; ++i) {
		for (size_t j = 0; j < IMAGE_WIDTH; ++j) {
			img[i][j] = (unsigned char)raw_img[i][j] / 255.0;
		}
	}

	close(fd);

	return label;
}

ssize_t parse_testing_data(struct image *data, size_t dlen)
{
	ssize_t ret;
	_Bool run;
	size_t num;

	char path[PATH_MAX];
	double img[IMAGE_HEIGHT][IMAGE_WIDTH];
	DIR *dir = NULL;
	struct dirent *ent = NULL;

	num = 0;
	memset(path, 0, sizeof(path));
	dir = opendir(TESTING_DATA_PATH);
	if (dir == NULL) {
		warn("opendir");
		return -1;
	}

	for (;;) {
		errno = 0;
		ent = readdir(dir);
		if (ent == NULL && errno == 0) {
			break;
		} else if (ent == NULL) {
			warn("readdir");
			return -1;
		}

		if (ent->d_type != DT_REG)
			continue;

		if (num == dlen)
			break;

		strcpy(path, TESTING_DATA_PATH);
		strcat(path, ent->d_name);

		ret = parse_image(img, path);
		if (ret < 0) {
			return ret;
		}

		data[num].label = ret;
		memcpy(data[num].img, img, sizeof(img));
		++num;
	}

	return 0;
}

ssize_t parse_training_data(struct image *data, size_t dlen)
{
	ssize_t ret;
	_Bool run;
	size_t num;

	char path[PATH_MAX];
	double img[IMAGE_HEIGHT][IMAGE_WIDTH];
	DIR *dir = NULL;
	struct dirent *ent = NULL;

	num = 0;
	memset(path, 0, sizeof(path));
	dir = opendir(TRAINING_DATA_PATH);
	if (dir == NULL) {
		warn("opendir");
		return -1;
	}

	for (;;) {
		errno = 0;
		ent = readdir(dir);
		if (ent == NULL && errno == 0) {
			break;
		} else if (ent == NULL) {
			warn("readdir");
			return -1;
		}

		if (ent->d_type != DT_REG)
			continue;

		if (num == dlen)
			break;

		strcpy(path, TRAINING_DATA_PATH);
		strcat(path, ent->d_name);

		ret = parse_image(img, path);
		if (ret < 0) {
			return ret;
		}

		data[num].label = ret;
		memcpy(data[num].img, img, sizeof(img));
		++num;
	}

	return 0;
}
