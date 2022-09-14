ARCH=native

access_n.exe: access_n.cu
	@git log -1 --format=format:"%H" > .COMPILE_ENV
	@nvcc -arch=$(ARCH) access_n.cu -o access_n.exe -lm

.PHONY: clean
clean:
	rm -f access_n.exe
