main: main.c culzss  gpu_compress
	gcc -g -L /usr/local/cuda/lib64/ -o main main.c culzss.o gpu_compress.o -lcudart -lpthread -Wl,-rpath=/usr/local/cuda/lib64       

culzss:  culzss.c culzss.h 
	gcc -g  -c -lpthread -o culzss.o culzss.c

gpu_compress: gpu_compress.cu gpu_compress.h
	nvcc -O3 -g -c -arch sm_20  -lpthread -o gpu_compress.o gpu_compress.cu      

clean:
	rm *.o
	rm *.gch
