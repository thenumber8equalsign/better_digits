/* test the images generated */
#include <config.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <err.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <fcntl.h>

int main(void)
{
	ssize_t ret;
	char path[PATH_MAX + 54];
	unsigned char data[28 * 28 + 50];
	int fd;
	_Bool run;

	struct dirent *ent = NULL;
	DIR *dir = NULL;

	memset(path, 0, sizeof(path));
	memset(data, 0, sizeof(data));

	ret = chdir("../data/training_data/");
	if (ret != 0)
		err(EXIT_FAILURE, "chdir");

	dir = opendir(".");
	if (dir == NULL)
		err(EXIT_FAILURE, "opendir");

	do {
		errno = 0;
		ent = readdir(dir);
		if (ent == NULL && errno != 0)
			err(EXIT_FAILURE, "readdir");
		else if (ent == NULL) {
			fprintf(stderr, "idk\n");
			exit(EXIT_FAILURE);
		}

		const char *name = ent->d_name;
		run = name[strlen(name) - 1] != '9';
	} while (run);

	strcpy(path, ent->d_name);

	closedir(dir);
	dir = NULL;
	ent = NULL;

	fd = open(path, O_RDONLY);
	if (fd == -1)
		err(EXIT_FAILURE, "open");

	ret = lseek(fd, 24, SEEK_SET);
	if (ret == -1)
		err(EXIT_FAILURE, "lseek");

	ret = read(fd, data, 28 * 28);
	if (ret == -1)
		err(EXIT_FAILURE, "read");
	if (ret != 28 * 28) {
		fprintf(stderr, "idk (read)\n");
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < 28; ++i) {
		for (size_t j = 0; j < 28; ++j) {
			printf("%3d ", (int)data[i * 28 + j]);
		}
		printf("\n");
	}

	close(fd);
	return 0;
}
