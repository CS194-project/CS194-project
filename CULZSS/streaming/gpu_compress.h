/***************************************************************************
*          CUDA LZSS 
*
*   Purpose : Using  CUDA based compression functionality to test compression/decompression 
*   Authors  : Adnan Ozsoy, Martin Swany, Arun Chauhan,Indiana University - Bloomington
*   Update	: December 2012 (last update date)

****************************************************************************
	Copyright 2012 Adnan Ozsoy, Martin Swany, Arun Chauhan

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
	
****************************************************************************/
#ifndef __GPU_COMPRESS_H_
#define __GPU_COMPRESS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/***************************************************************************
*                                CONSTANTS
***************************************************************************/
#define FALSE   0
#define TRUE    1

#define WINDOW_SIZE 128

/* maximum match length not encoded and encoded (4 bits) */
#define MAX_UNCODED     2
#define MAX_CODED       128
#define NWORKERS	2

#define PCKTSIZE 4096

/***************************************************************************
*                            TYPE DEFINITIONS
***************************************************************************/
/* unpacked encoded offset and length, gets packed into 12 bits and 4 bits*/
typedef struct encoded_string_t
{
    int offset;     /* offset to start of longest match */
    int length;     /* length of longest match */
} encoded_string_t;

typedef enum
{
    ENCODE,
    DECODE
} MODES;

typedef struct aftercompdata
{
	int tid;
	int * header;     /* offset to start of longest match */
	unsigned char * buffer;
	int buf_length;
	unsigned char * bufferout;
	int numts;	
	int comptookmore;
	int newlen;
} aftercompdata_t;

struct thread_data{
   int  tid;
   int  numthreads;
   unsigned char *buf;
   unsigned char *outbuf;
   int * header;
   int msg_length;
   int success;
   unsigned char * in_d;
   int ** ledger;
   float timings;
   int nstreams;
   int * sizes;
};




/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

extern "C" int  compression_kernel_wrapper(unsigned char *buffer, int buf_length,unsigned char * compressed_buffer, int compression_type, int wsize, int numthre, int nstreams, int index,unsigned char * in_d,unsigned char * out_d);
extern "C" void  decompression_kernel_wrapper(unsigned char *buffer, int buf_length, int * comp_length, int compression_type, int wsize, int numthre,int index);
extern "C" int aftercompression_wrapper(unsigned char * buffer, int buf_length, unsigned char * bufferout, int * comp_length);
extern "C" unsigned char * initGPUmem( int buf_length);
extern "C" unsigned char * initCPUmem( int buf_length);
extern "C" void deleteGPUmem(unsigned char * mem_d);
extern "C" void deleteCPUmem(unsigned char * mem_d);
extern "C" void initGPU();
extern "C" void resetGPU();
extern "C" int streams_in_GPU();
extern "C" int onestream_finish_GPU(int index);
extern "C" void deleteGPUStreams();
extern "C" void signalExitThreads();

#endif

