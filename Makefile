.PHONY: all clean _build_dir _src

export CC = gcc
export CFLAGS = -std=c99 -O2
export INCLUDE := $(abspath ./include/)
export BUILD_DIR := $(abspath ./build/)
export RULES := $(abspath rule.mk)

EXE := $(BUILD_DIR)/recognize

include $(RULES)

all: _build_dir _src
	$(CC) $(BUILD_DIR)/*.o -o $(EXE)

clean:
	rm -r $(BUILD_DIR)

_src:
	make -C src/ all

_build_dir:
	mkdir -p $(BUILD_DIR)
