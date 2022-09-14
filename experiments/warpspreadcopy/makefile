# make warpspreadcopy

ARCH=native

warpspreadcopy.exe: warpspreadcopy.cu
	@nvcc -arch=$(ARCH) warpspreadcopy.cu -o warpspreadcopy.exe -lm

.PHONY: clean
clean:
	@rm warpspreadcopy.exe

# kept consistent using .exe extention
# note that linux doesn't need exe extention
