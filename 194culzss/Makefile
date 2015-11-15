############################################################################
# Makefile for lzss encode/decode library and sample program
############################################################################
CC = gcc
LD = gcc
CFLAGS = -I. -O3 -Wall -Wextra -pedantic -ansi -c
LDFLAGS = -O3 -o

# libraries
LIBS = -L. -llzss -loptlist

# Treat NT and non-NT windows the same
ifeq ($(OS),Windows_NT)
	OS = Windows
endif

ifeq ($(OS),Windows)
	ifeq ($(OSTYPE), cygwin)
		EXE = .exe
		DEL = rm
	else
		EXE = .exe
		DEL = del
	endif
else	#assume Linux/Unix
	EXE =
	DEL = rm -f
endif

# define the method to be used for searching for matches (choose one)
# brute force
# FMOBJ = brute.o

# linked list
# FMOBJ = list.o

# hash table
 FMOBJ = hash.o

# Knuth–Morris–Pratt search
# FMOBJ = kmp.o

# binary tree
#FMOBJ = tree.o

LZOBJS = $(FMOBJ) lzss.o

all:		sample$(EXE) liblzss.a liboptlist.a culzss

test: all
	./sample -c -i sample.c -o sample.gz
	./sample -d -i sample.gz -o sample2.c
	diff sample.c sample2.c
	ls -n sample.c
	ls -n sample.gz
	rm -f sample.gz sample2.c

sample$(EXE):	sample.o liblzss.a liboptlist.a
		$(LD) $< $(LIBS) $(LDFLAGS) $@

sample.o:	sample.c lzss.h optlist.h
		$(CC) $(CFLAGS) $<

liblzss.a:	$(LZOBJS) bitfile.o
		ar crv liblzss.a $(LZOBJS) bitfile.o
		ranlib liblzss.a

lzss.o:	lzss.c lzlocal.h bitfile.h
		$(CC) $(CFLAGS) $<

brute.o:	brute.c lzlocal.h
		$(CC) $(CFLAGS) $<

list.o:		list.c lzlocal.h
		$(CC) $(CFLAGS) $<

hash.o:		hash.c lzlocal.h
		$(CC) $(CFLAGS) $<

kmp.o:		kmp.c lzlocal.h
		$(CC) $(CFLAGS) $<

tree.o:		tree.c lzlocal.h
		$(CC) $(CFLAGS) $<

bitfile.o:	bitfile.c bitfile.h
		$(CC) $(CFLAGS) $<

liboptlist.a:	optlist.o
		ar crv liboptlist.a optlist.o
		ranlib liboptlist.a

optlist.o:	optlist.c optlist.h
		$(CC) $(CFLAGS) $<

clean:
		$(DEL) *.o
		$(DEL) *.a
		$(DEL) sample$(EXE)
		$(DEL) culzss
		$(DEL) *.gz
		$(DEL) reference

culzss: culzss.cu bitfile.c bitfile.h optlist.h optlist.c
	nvcc -O3 -arch=sm_30 culzss.cu -c -o culzss.o
	g++  -O3 bitfile.c -c -o bitfile2.o
	g++  -O3 optlist.c -c -o optlist2.o
	g++  -O3 bitfile2.o optlist2.o culzss.o -o culzss -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart

.PHONY: check
check: culzss sample
	/usr/bin/time ./culzss -i ../corpus/combined -o output.gz
	./sample -d -i output.gz -o reference
	ls -n ../corpus/combined
	ls -n output.gz
	diff ../corpus/combined reference
