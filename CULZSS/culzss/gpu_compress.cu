/***************************************************************************
*          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding on CUDA
*
*
****************************************************************************
*          CUDA LZSS 
*   Authors  : Adnan Ozsoy, Martin Swany,Indiana University - Bloomington
*   Date    : April 11, 2011

****************************************************************************

	Copyright 2011 Adnan Ozsoy, Martin Swany, Indiana University - Bloomington

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

/***************************************************************************
* Code is adopted from below source
*
* LZSS: An ANSI C LZss Encoding/Decoding Routine
* Copyright (C) 2003 by Michael Dipperstein (mdipper@cs.ucsb.edu)
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*
***************************************************************************/

/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "getopt.h"
#include <time.h>
#include "gpu_compress.h"
#include <pthread.h>


/***************************************************************************
*                             CUDA FILES
***************************************************************************/
#include <assert.h>
#include <cuda.h>
 

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

struct thread_data{
   int  tid;
   int  numthreads;
   unsigned char *buf;
   long msg_length;
   int success;
};


/***************************************************************************
*                                CONSTANTS
***************************************************************************/
#define FALSE   0
#define TRUE    1

//#define WINDOW_SIZE     1024   /* size of sliding window (12 bits) */
#define WINDOW_SIZE 128

/* maximum match length not encoded and encoded (4 bits) */
#define MAX_UNCODED     2
#define MAX_CODED       128

extern "C" int  compression_kernel_wrapper(unsigned char *buffer, long buf_length,unsigned char * compressed_buffer, long * comp_length, int compression_type, int wsize, int numthre);
extern "C" void  decompression_kernel_wrapper(unsigned char *buffer, long buf_length, long * comp_length, int compression_type, int wsize, int numthre);

unsigned char * decompressed_buffer; 

//#define SIZEBLOCK 4
#define PCKTSIZE 4096
//int numThreads = 256;
//#define NUMTHRE 512

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
//unsigned char slidingWindow[WINDOW_SIZE];
//unsigned char uncodedLookahead[MAX_CODED];

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/
//void EncodeLZSS(FILE *inFile, FILE *outFile);   /* encoding routine */
//void DecodeLZSS(FILE *inFile, FILE *outFile);   /* decoding routine */

/***************************************************************************
*                                FUNCTIONS
***************************************************************************/



/****************************************************************************
*   Function   : FindMatch
*   Description: This function will search through the slidingWindow
*                dictionary for the longest sequence matching the MAX_CODED
*                long string stored in uncodedLookahed.
*   Parameters : windowHead - head of sliding window
*                uncodedHead - head of uncoded lookahead buffer
*   Effects    : NONE
*   Returned   : The sliding window index where the match starts and the
*                length of the match.  If there is no match a length of
*                zero will be returned.
****************************************************************************/
__device__ encoded_string_t FindMatch(int windowHead, int uncodedHead, unsigned char* slidingWindow, unsigned char * uncodedLookahead, int tx, int wfilepoint, int lastcheck)
{
    encoded_string_t matchData;
    int i, j, k, l;
	int maxcheck;

    matchData.length = 1; // make it 1 in the 0 case, it will be returned as 1, 0 gives problems
    i = windowHead ;  /* start at the beginning of the sliding window */
    j = 0;
	k =0;
	l=0;
	
	if(lastcheck) 
		maxcheck = MAX_CODED - tx;
	else
		maxcheck = MAX_CODED;

		while (TRUE)
		{
			//to access memory all together
			//__syncthreads();


			if (slidingWindow[i] == uncodedLookahead[uncodedHead])
			{
				/* we matched one how many more match? */
				j = 1;
				l = k;
				
				if (l >= maxcheck-1)
					{
						/* we wrapped around */
						break;
					}
					
				while(slidingWindow[(i + j) % (WINDOW_SIZE+MAX_CODED)] ==
					uncodedLookahead[(uncodedHead + j)% (MAX_CODED*2)])
				{
				
					if (j >= maxcheck)
					{
						break;
					}
					
					j++;
					l++;
			
					if (l >= maxcheck-1)
					{
						/* we wrapped around */
						break;
					}
				}

				if (j > matchData.length)
				{
					matchData.length = j;
					matchData.offset = i;
				}
			}
			if (j >= maxcheck)
			{
				matchData.length = MAX_CODED;
				break;
			}
			
			k = k + 1;
			i = (i + 1) % (WINDOW_SIZE+MAX_CODED);
			
			if (k == maxcheck)
			{
				/* we wrapped around */
				break;
			}

		}
		
    return matchData;
}

void checkCUDAError(const char *msg)
{
 cudaError_t err = cudaGetLastError();
 if( cudaSuccess != err) 
 {
  fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
 cudaGetErrorString( err) );
  exit(EXIT_FAILURE);
 } 
}


__global__ void EncodeKernel(unsigned char * in_d, unsigned char * out_d, int SIZEBLOCK)
{


	/* cyclic buffer sliding window of already read characters */
	__shared__ unsigned char slidingWindow[WINDOW_SIZE+(MAX_CODED)];
	__shared__ unsigned char uncodedLookahead[MAX_CODED*2];

	__shared__ unsigned char encodedData[MAX_CODED*2];
    	encoded_string_t matchData;

	int windowHead, uncodedHead;    // head of sliding window and lookahead //
	int filepoint;			//file index pointer for reading
	int wfilepoint;			//file index pointer for writing
	int lastcheck;			//flag for last run of the packet

	int bx = blockIdx.x;
	int tx = threadIdx.x; 
	
	
   //***********************************************************************
   // * Fill the sliding window buffer with some known vales.  DecodeLZSS must
   // * use the same values.  If common characters are used, there's an
   // * increased chance of matching to the earlier strings.
   // *********************************************************************** //

	slidingWindow[tx] = ' ';
	windowHead = tx;
	uncodedHead = tx;	
	filepoint=0;
	wfilepoint=0;
	lastcheck=0;
	
	__syncthreads();

	
	//***********************************************************************
	//* Copy MAX_CODED bytes from the input file into the uncoded lookahead
	//* buffer.
	//*********************************************************************** //
  
	uncodedLookahead[tx] = in_d[bx * PCKTSIZE + tx];
	filepoint+=MAX_CODED;
	
	slidingWindow[ (windowHead + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[uncodedHead];
	
	__syncthreads(); 
	
	uncodedLookahead[MAX_CODED+tx] = in_d[bx * PCKTSIZE + filepoint + tx];
	filepoint+=MAX_CODED;
	

	
	__syncthreads();

    
	// Look for matching string in sliding window //	
	matchData = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead, tx,0, 0);
	__syncthreads();  
	
	// now encoded the rest of the file until an EOF is read //
	while ((filepoint) <= PCKTSIZE)
	{		
	
		if(lastcheck==1)
		{
			if(matchData.length > (MAX_CODED - tx))
				matchData.length = MAX_CODED - tx;
		}
		
		if (matchData.length >= MAX_CODED)
        {
			// garbage beyond last data happened to extend match length //
			matchData.length = MAX_CODED-1;
		}

		if (matchData.length <= MAX_UNCODED)
		{
			// not long enough match.  write uncoded byte //
			matchData.length = 1;   // set to 1 for 1 byte uncoded //
			encodedData[tx*2] = 1;
			encodedData[tx*2 + 1] = uncodedLookahead[uncodedHead];
		}
		else if(matchData.length > MAX_UNCODED)
		{	
			// match length > MAX_UNCODED.  Encode as offset and length. //
			encodedData[tx*2] = (unsigned char)matchData.length;
			encodedData[tx*2+1] = (unsigned char)matchData.offset;			
		}
		
		//write out the encoded data into output
		out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2] = encodedData[tx*2];
		out_d[bx * PCKTSIZE*2 + wfilepoint + tx*2 + 1] = encodedData[tx*2+1];
		
		//update written pointer and heads
		wfilepoint = wfilepoint + MAX_CODED*2;
		
		windowHead = (windowHead + MAX_CODED) % (WINDOW_SIZE+MAX_CODED);
		uncodedHead = (uncodedHead + MAX_CODED) % (MAX_CODED*2);
		
		__syncthreads(); 	

				
		if(lastcheck==1)
		{
			break;			
		}	
		
		
		if(filepoint<PCKTSIZE){
			uncodedLookahead[(uncodedHead+ MAX_CODED)% (MAX_CODED*2)] = in_d[bx * PCKTSIZE + filepoint + tx];
			filepoint+=MAX_CODED;
			
			//find the location for the thread specific view of window
			slidingWindow[ (windowHead + WINDOW_SIZE ) % (WINDOW_SIZE + MAX_CODED) ] = uncodedLookahead[uncodedHead];
			//__syncthreads(); 	
		}
		else{
			lastcheck++;				
			slidingWindow[(windowHead + MAX_CODED ) % (WINDOW_SIZE+MAX_CODED)] = '^';		
		}
		__syncthreads(); 	
		

		matchData = FindMatch(windowHead, uncodedHead,slidingWindow,uncodedLookahead,tx,wfilepoint, lastcheck);

	} //while

}


void * thread_func(void * t_struc)
//void * thread_func(void * numT)
{
	struct thread_data * t_data=(struct thread_data *)t_struc;
	
	unsigned char * buffer = t_data->buf;
	long buf_length=t_data->msg_length;
	int devnum=t_data->tid;
	 
	int comptookmore = 0;
	
	//Calculate padding and update the size but not add yet, will add padding directly to gpu buffer
	int paddingsize=(PCKTSIZE - (buf_length%PCKTSIZE))%PCKTSIZE;
	int buf_length_padded=buf_length;
	unsigned char * buffer_pad;
	//printf("padsize %d buf_length %ld \n",paddingsize, buf_length);
	if(paddingsize!=0)
	{
		buf_length_padded += paddingsize;
		buffer_pad = (unsigned char *)malloc(sizeof(char)*paddingsize);
		for(int i=0;i<paddingsize;i++)
			buffer_pad[i]=0;
	}
	unsigned char * bufferout;
	bufferout = (unsigned char *)malloc(sizeof(char)*buf_length_padded*2);
	
	int numThreads = t_data->numthreads;
	int numblocks;
	
	clock_t start = clock();
	clock_t memkermem = clock();
		
	numblocks = buf_length_padded / (PCKTSIZE);

	cudaDeviceProp props;
	unsigned char * in_d;
	unsigned char * out_d;

	cudaSetDevice(devnum); // explicit set the current device for the other calls
	
	int i;
    cudaGetDeviceProperties(&props, devnum);

	//time the process	
	start = clock();

	//Allocate cuda memory
	cudaMalloc((void **) &in_d, sizeof(char)*buf_length_padded);
	cudaMalloc((void **) &out_d, sizeof(char)*buf_length_padded*2);

	//printf("First mem alloc time %f \n", ((double)clock()-start)/CLOCKS_PER_SEC);
	start = clock();
	memkermem = clock();
	//copy memory to cuda device
	cudaMemcpy(in_d, buffer, sizeof(char)*buf_length,cudaMemcpyHostToDevice);
	//copy padding
	if(paddingsize!=0)
		cudaMemcpy(in_d+buf_length, buffer_pad, sizeof(char)*paddingsize,cudaMemcpyHostToDevice);	
	checkCUDAError("mem copy to gpu");
	
	//printf("First mem copy time %f \n", ((double)clock()-start)/CLOCKS_PER_SEC);

	start = clock();
	//compression kernel
	EncodeKernel<<< numblocks, numThreads >>>(in_d,out_d,numThreads);
	// Check for any CUDA errors
	checkCUDAError("kernel invocation");
	 
	cudaThreadSynchronize();
	checkCUDAError("kernel invocation after sync");
	//printf("Only  compr kernel run %f \n", ((double)clock()-start)/CLOCKS_PER_SEC);
	start = clock();
	
	//copy memory back
	cudaMemcpy(bufferout, out_d, sizeof(char)*buf_length_padded*2, cudaMemcpyDeviceToHost);
	//cudaMemcpy(print_cpu, print_d, sizeof(char)*numThreads, cudaMemcpyDeviceToHost);
	checkCUDAError("mem copy2");

	//printf("Mem copy back time %f \n", ((double)clock()-start)/CLOCKS_PER_SEC);

	// ******************************************************************
	// * After cuda kernel processing-
	// * The data given from GPUs is in a two dimension array, where first number shows how many matching
	// * and the second shows either the character or the location of matching start for matchings more than 1
	// * So after process eliminates the redundant ones from this array.
	// ******************************************************************
	start = clock();
	
	int j, k, temptot, tempj;
	unsigned char flags;
	unsigned char flagPos;
	
	// allocate memory to contain the header of the file:
	int * header;
	header = (int *)malloc (sizeof(int)*(buf_length_padded/4096));
	if (header == NULL) {printf ("Memory error"); exit (2);}

 	flags = 0;
	flagPos = 0x01;
	tempj=0;
	
	unsigned char holdbuf[16];
	int holdbufcount=0;
	int m=0;

	int morecounter=0;	
	//move data from double sized array to file buffer 
	//   by eliminating extra compression info
	for(k=0,i=0,j=0;i<buf_length_padded*2;)
	{
		if (j>buf_length) { 
			printf("compression took more!!! \n"); 
			comptookmore = 1;
			break;
		}
		temptot = bufferout[i];
		if(temptot == 1) //if no matching
		{
			flags |= flagPos;       // mark with uncoded byte flag //
			holdbuf[holdbufcount]=bufferout[i+1];
			holdbufcount++;
			i=i+2;
		}		
		else //if there is mathcing
		{
			holdbuf[holdbufcount]=temptot;
			holdbufcount++;
			holdbuf[holdbufcount]=bufferout[i+1];
			holdbufcount++;
			i=i+(temptot*2);
		}
				
		if (flagPos == 0x80) //if we have looked at 8 characters that fills the flag holder
		{
			buffer[j] = flags;
			j++;
			
			for(m=0;m<holdbufcount;m++){
				buffer[j] = holdbuf[m];
				j++;
			}
						
			// reset encoded data buffer //
			flags = 0;
			flagPos = 0x01;
			holdbufcount=0;
		}
		else
		{
			// we don't have 8 code flags yet, use next bit for next flag //
			flagPos <<= 1;
		}

		// for each packet with the size of 4096 bytes
		if(i%8192 == 0 && i>0){
			if(holdbufcount>0){
				buffer[j] = flags;
				j++;
				
				for(m=0;m<holdbufcount;m++){
				buffer[j] = holdbuf[m];
				j++;
				}
				holdbufcount=0;
			}
			
			flags = 0;
			flagPos = 0x01;			
			if((j-tempj) >= PCKTSIZE){
					morecounter++;
					//compression took more, so just write the file without compression info
				}					

			header[k]=j-tempj;
			tempj=j;
			k++;
		}
	}
	if(!comptookmore){
		//Add header to buffer
		unsigned char cc;
		for(i=0;i<k;i++)
		{
			cc = (unsigned char)(header[i]>>8);
			buffer[j]=cc;
			j++;
			cc=(unsigned char)header[i];
			buffer[j]=cc;
			j++;
		}
		
		//Add total size
		cc = (unsigned char)(buf_length_padded>>24);
		buffer[j]=cc;
		j++;
		cc = (unsigned char)(buf_length_padded>>16);
		buffer[j]=cc;
		j++;
		cc = (unsigned char)(buf_length_padded>>8);
		buffer[j]=cc;
		j++;
		cc=(unsigned char)buf_length_padded;
		buffer[j]=cc;
		j++;
		
		//Add pad size
		cc = (unsigned char)(paddingsize>>8);
		buffer[j]=cc;
		j++;
		cc=(unsigned char)paddingsize;
		buffer[j]=cc;
		j++;
		
		//printf("padsize %d buf_length %d \n",paddingsize, buf_length_padded);
		
		t_data->msg_length = j;
	}
	
	printf("after kernel process took %f \n", ((double)clock()-start)/CLOCKS_PER_SEC);
	
	//free cuda memory
	cudaFree(in_d);	
	//cudaFree(print_d);
	cudaFree(out_d);
	//free memory
	free(bufferout);
	free(header);
	//free(print_cpu);
	
	return (void *)((long long)(comptookmore));
}

/****************************************************************************
*   Function   : EncodeLZSS
*   Description: This function will read an input buffer and write to the output
*                buffer encoded using a slight modification to the LZss
*                algorithm.  I'm not sure who to credit with the slight
*                modification to LZss, but the modification is to group the
*                coded/not coded flag into bytes.  By grouping the flags,
*                the need to be able to write anything other than a byte
*                may be avoided as longs as strings encode as a whole byte
*                multiple.  This algorithm encodes strings as 16 bits (a 12
*                bit offset + a 4 bit length).
****************************************************************************/
int compression_kernel_wrapper(unsigned char *buffer, long buf_length, unsigned char * compressed_buffer, long * comp_length, int compression_type,int wsize, int numthre)
{

	clock_t start = clock();
	clock_t memkermem = clock();
	clock_t memkermemwIO = clock();
	
	int i;
	int comptookmore=0;

	/* 8 code flags and encoded strings */
	int numThreads = numthre;
	
	int num_devices;
	cudaGetDeviceCount(&num_devices);
 	struct thread_data thread_data_array[num_devices];
	pthread_t threads[num_devices];
	void *status;
   
	for(i=0;i<num_devices; i++){

		long bsize = (buf_length/num_devices);
		
		thread_data_array[i].tid=i;
		thread_data_array[i].numthreads=numThreads;
		thread_data_array[i].buf=(buffer+(bsize*i));
		thread_data_array[i].msg_length=bsize;
		
		pthread_create( &threads[i], NULL, &thread_func, &thread_data_array[i] );
	
	}
	
	for(i=0;i<num_devices; i++){
		pthread_join( threads[i], &status);
		comptookmore += (int)((long long)status);
	}
	
	if(comptookmore!=0)
		return 0;
	
	long currentstrloc = thread_data_array[0].msg_length;
	
	for(i=0;i<currentstrloc;i++)
		compressed_buffer[i]=buffer[i];
	
	*comp_length = currentstrloc;

	return 1;

}

	
__global__ void DecodeKernel(unsigned char * in_d, unsigned char * out_d, int * error_d, int * sizearr_d, int SIZEBLOCK)
{

	// cyclic buffer sliding window of already read characters //
	unsigned char slidingWindow[WINDOW_SIZE];
	unsigned char uncodedLookahead[MAX_CODED];
	//unsigned char writebuf[8];
	
    int nextChar;                       /* next char in sliding window */
    encoded_string_t code;              /* offset/length code for string */

    // 8 code flags and encoded strings //
    unsigned char flags, flagsUsed;
    int i, c;

     // // initialize variables * /
    flags = 0;
    flagsUsed = 7;
    nextChar = 0;
	
	int filepoint=0;
	int wfilepoint=0;

	int bx = blockIdx.x;
	int tx = threadIdx.x; 
	
	int sizeinpckt = 0, startadd = 0;
	startadd = sizearr_d[bx * SIZEBLOCK + tx]; //read the size of the packet
	sizeinpckt = sizearr_d[bx * SIZEBLOCK + tx + 1] -  startadd;
 
	//bigger than a packet hold-compression took more space for that packet
	//REPORT 
	if(sizeinpckt > PCKTSIZE){
		(*error_d)++;
	}

	// ************************************************************************
	//* Fill the sliding window buffer with some known vales.  EncodeLZSS must
	//* use the same values.  If common characters are used, there's an
	//* increased chance of matching to the earlier strings.
	//************************************************************************ /
	for (i = 0; i < WINDOW_SIZE; i++)
	{
		slidingWindow[i] = ' ';
	}

	while (TRUE)
		{
			flags >>= 1;
			flagsUsed++;

			if (flagsUsed == 8)
			{
				// shifted out all the flag bits, read a new flag //
				if (filepoint >= sizeinpckt)
				{
					break;
				}
				c=in_d[startadd + filepoint]; //packet*PCKTSIZE 
				filepoint++;
				flags = c & 0xFF;
				flagsUsed = 0;
			}

			if (flags & 0x01)
			{
				// uncoded character //
				if (filepoint >= sizeinpckt)
				{
					break;
				}
				
				// write out byte and put it in sliding window putc(c, outFile);//
				out_d[bx * SIZEBLOCK * PCKTSIZE + tx*PCKTSIZE+wfilepoint]=in_d[startadd +filepoint];
				wfilepoint++;
				slidingWindow[nextChar] = in_d[startadd +filepoint];
				nextChar = (nextChar + 1) % WINDOW_SIZE;
				filepoint++;
			}
			else
			{
				// offset and length //
				if (filepoint >= sizeinpckt)
				{
					break;
				}
				code.length=in_d[startadd +filepoint];
				filepoint++;

				if (filepoint >= sizeinpckt)
				{
					break;
				}
				code.offset =in_d[startadd +filepoint];
				filepoint++;
				
				// ****************************************************************
				//* Write out decoded string to file and lookahead.  It would be
				//* nice to write to the sliding window instead of the lookahead,
				////* but we could end up overwriting the matching string with the
				///* new string if abs(offset - next char) < match length.
				//**************************************************************** /
				for (i = 0; i < code.length; i++)
				{
					c = slidingWindow[(code.offset + i) % WINDOW_SIZE];
					out_d[bx * SIZEBLOCK * PCKTSIZE + tx*PCKTSIZE + wfilepoint]=c;
					wfilepoint++;
					//putc(c, outFile);//
					uncodedLookahead[i] = c;
				}

				// write out decoded string to sliding window //
				for (i = 0; i < code.length; i++)
				{
					slidingWindow[(nextChar + i) % WINDOW_SIZE] =
						uncodedLookahead[i];
				}

				nextChar = (nextChar + code.length) % WINDOW_SIZE;
			}
		}
		
}
	

void decompression_kernel_wrapper(unsigned char *buffer, long buf_length, long * decomp_length, int compression_type,int wsize, int numthre)
{

	int i,j;
	int numblocks;
	int devID;
	unsigned char * in_d, * out_d;	
	cudaDeviceProp props;
	int numThreads=numthre;

	//get original file size from compressed file
	int origsize=0;
	origsize = buffer[buf_length-6] << 24;
	origsize = origsize  ^ ( buffer[buf_length-5]<< 16);
	origsize = origsize  ^ ( buffer[buf_length-4]<< 8);
	origsize = origsize  ^ ( buffer[buf_length-3]);
	long lSize =  origsize;

	//padsize
	int padsize=0;
	padsize = ( buffer[buf_length-2]<< 8);
	padsize = padsize  ^ ( buffer[buf_length-1]);

	printf("orig size: %d %d\n",origsize,padsize);	
	numblocks =lSize / ((numThreads)*PCKTSIZE);
	
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
	
	// put the bottom size headers into an array
	int  * sizearr, * sizearr_d;
	sizearr = (int *)malloc(sizeof(int)*(numThreads*numblocks+1));
	if (sizearr == NULL) {printf ("Memory error"); exit (2);}
	
	decompressed_buffer = (unsigned char*) malloc (sizeof(char)*(lSize-padsize));
	if (decompressed_buffer == NULL) {printf ("Memory error in decompression"); exit (2);}
	
	sizearr[0]=0;
	int temptot=0;
	int total=0;
	for(i=1, j=0;i<(numThreads*numblocks+1);i++, j=j+2)
	{
		temptot = buffer[buf_length-2*numThreads*numblocks+j-6] << 8;
		temptot = temptot  ^ ( buffer[buf_length-2*numThreads*numblocks+j+1-6]);

		total = total + temptot;
		sizearr[i]=total;	
	}
	
	int * error_c;
	error_c = (int *)malloc(sizeof(int));
	if (error_c==NULL) exit (1);
	
	*error_c = 0;
	
	int * error_d;
	
	cudaMalloc((void **) &error_d, sizeof(int));
	
	cudaMemcpy(error_d, error_c, sizeof(int),cudaMemcpyHostToDevice);
	checkCUDAError("mem copy0");

	//Allocate cuda memory
	cudaMalloc((void **) &in_d, sizeof(char)*(buf_length-2*numThreads*numblocks-6));
	cudaMalloc((void **) &out_d, sizeof(char)*lSize);
	cudaMalloc((void **) &sizearr_d, sizeof(int)*(numThreads*numblocks+1));	

	//copy memory to cuda
	cudaMemcpy(in_d, buffer, sizeof(char)*(buf_length-2*numThreads*numblocks-6),cudaMemcpyHostToDevice);
	cudaMemcpy(sizearr_d, sizearr, sizeof(int)*(numThreads*numblocks+1),cudaMemcpyHostToDevice);
	checkCUDAError("mem copy1");

	//decompression kernel
	DecodeKernel<<< numblocks, numThreads >>>(in_d,out_d,error_d,sizearr_d,numThreads);
	
	// Check for any CUDA errors
	checkCUDAError("kernel invocation");
	
	//copy memory back
	cudaMemcpy(decompressed_buffer, out_d, sizeof(char)*(lSize-padsize), cudaMemcpyDeviceToHost);
	cudaMemcpy(error_c, error_d, sizeof(int),cudaMemcpyDeviceToHost);
	
	checkCUDAError("mem copy2");
	
	if(*error_c !=0)
		printf("Compression took more space for some packets !!! code: %d \n", *error_c);

	* decomp_length = lSize-padsize;
		
	//free cuda memory
	cudaFree(in_d);
	cudaFree(error_d);
	cudaFree(out_d);
	cudaFree(sizearr_d);
	free(error_c);
 
}


