#!/usr/bin/env python
from tensorflow.keras.datasets import mnist
import struct

(train_image, train_label), (test_image, test_label) = mnist.load_data()

for i in range(len(train_image)):
	with open(f"training_data/{i}-{train_label[i]}.bin", "wb") as binary_file:
		binary_file.write(struct.pack("<B", train_label[i]))
		binary_file.write(
			b"\x1c\x1c"
		)
		for j in range(len(train_image[i])):
			for k in range(len(train_image[i][j])):
				binary_file.write(struct.pack("<B", train_image[i][j][k]))

for i in range(len(test_image)):
	with open(f"testing_data/{i}-{test_label[i]}.bin", "wb") as binary_file:
		binary_file.write(struct.pack("<B", test_label[i]))
		binary_file.write(b"\x1c\x1c")
		for j in range(len(test_image[i])):
			for k in range(len(test_image[i][j])):
				binary_file.write(struct.pack("<B", test_image[i][j][k]))
