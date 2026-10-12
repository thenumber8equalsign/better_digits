$(BUILD_DIR)/%.c.o: %.c
	$(CC) -c $(CFLAGS) -I $(INCLUDE) $< -o $@
