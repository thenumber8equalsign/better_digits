#!/usr/bin/env python

# All this has done is make me hate python even more
# WHY IS % FORMATTING DEPRECATED???? IT WORKS JUST FINE AND C PROGRAMMERS ARE
# USED TO IT
# WHY IS IT CALLED "except" INSTEAD OF CATCH???? (python does predate java so I
# will give this one a slide however)
# I HATE TRY CATCH, ALL IT DOES IT MAKE IT HARD TO IGNORE ERRORS
# RETURNING A RETURN CODE IS OBVIOUSLY THE BETTER CHOICE CUZ IT CAN BE IGNORED
# EASILY, AND IF YOU COMPLAIN ABOUT "but what if you don"t know that it can
# error," then you should READ THE DOCUMENTATION FOR A FUNCTION BEFORE CALLING
# IT.

import numpy
import os
import struct

import scipy.io as sio

mat = sio.loadmat("emnist/emnist-digits.mat")
data = mat["dataset"]

labels = data["train"][0, 0]["labels"][0, 0]
images = data["train"][0, 0]["images"][0, 0]
images = images.reshape((images.shape[0], 28, 28), order="F")

"""
for i in range(28):
	for j in range(28):
		print("{:3} ".format(images[0][i][j]), sep="", end="")
	print()
"""

try:
	os.mkdir("data")
except FileExistsError:
	pass

try:
	os.mkdir("data/training_data")
except FileExistsError:
	pass

for i in range(len(images)):
	file = open(f"data/training_data/{i}-{labels[i][0]}", "wb")
	file.write(struct.pack("<B", labels[i][0] + 0x30))
	file.write(struct.pack("<BBBBBBB", 0, 0, 0, 0, 0, 0, 0))
	file.write(struct.pack("<QQ", 28, 28))
	arr = images[i].flatten(order="C").tobytes(order="C")
	file.write(arr)
	file.close()

try:
	os.mkdir("data/testing_data")
except FileExistsError:
	pass

labels = data["test"][0, 0]["labels"][0, 0]
images = data["test"][0, 0]["images"][0, 0]
images = images.reshape((images.shape[0], 28, 28), order="F")

for i in range(len(images)):
	file = open(f"data/testing_data/{i}-{labels[i][0]}", "wb")
	file.write(struct.pack("<B", labels[i][0] + 0x30))
	file.write(struct.pack("<BBBBBBB", 0, 0, 0, 0, 0, 0, 0))
	file.write(struct.pack("<QQ", 28, 28))
	arr = images[i].flatten(order="C").tobytes(order="C")
	file.write(arr)
	file.close()
