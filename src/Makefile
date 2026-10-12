.PHONY: all _libs
include $(RULES)

DEPS = $(BUILD_DIR)/main.c.o

all: _libs $(DEPS)

_libs:
	make -C libs
