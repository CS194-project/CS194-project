/***************************************************************************
*          CUDA LZSS 
*
*   Purpose : Using  CUDA based compression functionality to test compression/decompression 
*   Authors  : Adnan Ozsoy, Martin Swany, Arun Chauhan,Indiana University - Bloomington
*   Update	: December 2012 (last update date)

****************************************************************************
	Copyright 2012 Adnan Ozsoy, Martin Swany, Arun Chauhan,Indiana University - Bloomington

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
	
***************************************************************************/

#ifndef CULZSS_H
#define CULZSS_H

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

typedef struct {
	unsigned char ** buf;
	unsigned char ** bufout;
	
	unsigned char * in_d;	
	unsigned char * out_d;	
	
	int headPG;
	int headGC;
	int headSC;
	int headCS;
	int headSP;
	int * ledger;
	
	int * outsize;
	//int full, empty;
	pthread_mutex_t *mut;
	pthread_cond_t *produced, *compressed, *streamready, *sendready, *sent;
} queue;


//gpu functions
extern int  compression_kernel_wrapper(unsigned char * buffer, int buf_length,unsigned char * compressed_buffer, int compression_type, int wsize, int numthre, int nstreams, int index,unsigned char * in_d,unsigned char * out_d);
extern int  decompression_kernel_wrapper(unsigned char * buffer, int buf_length,unsigned char * decompressed_buffer, int * comp_length, int compression_type, int wsize, int numthre);
extern int aftercompression_wrapper(unsigned char * buffer, int buf_length, unsigned char * bufferout, int * comp_length);
extern unsigned char * initGPUmem( int buf_length);
extern unsigned char * initCPUmem( int buf_length);
extern void deleteGPUmem(unsigned char * mem_d);
extern void deleteCPUmem(unsigned char * mem_d);
extern void initGPU();
extern void resetGPU();
extern int streams_in_GPU();
extern int onestream_finish_GPU(int index);
extern void deleteGPUStreams();
extern void signalExitThreads();

//Queue functions
queue *queueInit (int maxiterations,int numblocks,int blocksize);
void queueDelete (queue *q);
void queueAdd (queue *q, int in);

void init_compression(queue *q,int maxiterations,int numblocks,int blocksize);
void join_comp_threads();

int getloopcount();

#endif
