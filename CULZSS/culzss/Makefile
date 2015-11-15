test: gpu_compress_wrapper.o gpu_compress.o test.o	
	gcc -g -L /usr/local/cuda/lib64/ -o test   gpu_compress_wrapper.o gpu_compress.o test.o -Wl,-rpath=/usr/local/cuda/lib64 -lcudart -lpthread

test.o: test.c 
	gcc -c -g  test.c -o test.o

phoebus_gpu_compress.o:  gpu_compress_wrapper.c gpu_compress.h
	gcc -I -g /usr/local/cuda/include -c -o  gpu_compress_wrapper.o  gpu_compress_wrapper.c	

gpu_compress.o: gpu_compress.cu gpu_compress.h
	nvcc -c -g   -lpthread -o gpu_compress.o gpu_compress.cu	

clean: 
	rm *.o
