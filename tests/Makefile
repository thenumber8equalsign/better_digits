.PHONY: matrix image clean libs _build_dir

export CFLAGS := -O2 -std=c99
export CC := gcc
export INCLUDE := $(abspath ../include/)
export BUILD_DIR := $(abspath ./build/)
export RULES := $(abspath ../rule.mk)
export LINK := -lm

include $(RULES)

matrix: clean _build_dir libs $(BUILD_DIR)/test_matrix.c.o
	$(CC) -o matrix.out $(BUILD_DIR)/*.o $(LINK) $(CFLAGS)

image: clean _build_dir libs $(BUILD_DIR)/test_image.c.o
	$(CC) -o image.out $(BUILD_DIR)/*.o $(LINK) $(CFLAGS)

clean:
	-rm -r $(BUILD_DIR)
	-rm -r matrix.out
	-rm -r image.out

libs:
	make -C ../src/libs/

_build_dir:
	mkdir -p $(BUILD_DIR)
