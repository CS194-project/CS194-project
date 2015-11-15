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
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "culzss.h"
#include <sys/time.h>

pthread_t congpu, constr, concpu, consend;

int loopnum=0;
int maxiterations=0;
int numblocks=0;
int blocksize=0;

int exit_signal = 0;


int getloopcount(){
	return loopnum;
}

void *gpu_consumer (void *q)
{

	struct timeval t1_start,t1_end,t2_start,t2_end;
	double time_d, alltime;

	queue *fifo;
	int i, d;
	int success=0;
	fifo = (queue *)q;
	int comp_length=0;
	
	fifo->in_d = initGPUmem((int)blocksize);
	fifo->out_d = initGPUmem((int)blocksize*2);
	
	
	for (i = 0; i < maxiterations; i++) {
		
		if(exit_signal){
			exit_signal++;
			break;
		}
		
		gettimeofday(&t1_start,0);	
	
		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headGC]!=1) 
		{
			pthread_cond_wait (fifo->produced, fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
		
		gettimeofday(&t2_start,0);	
		
		success=compression_kernel_wrapper(fifo->buf[fifo->headGC], blocksize, 
					fifo->bufout[fifo->headGC], 
					0, blocksize, 128, 1,fifo->headGC, fifo->in_d, fifo->out_d);
		if(!success){
		printf("Compression failed. Success %d\n",success);
		}	
		
		gettimeofday(&t2_end,0);
		time_d = (t2_end.tv_sec-t2_start.tv_sec) + (t2_end.tv_usec - t2_start.tv_usec)/1000000.0;
		printf("GPU kernel took:\t%f \t", time_d);
				
		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headGC]++;
		fifo->headGC++;
		if (fifo->headGC == numblocks)
			fifo->headGC = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->compressed);
		
		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		printf("GPU whole took:\t%f \n", alltime);
	}
	
	deleteGPUmem(fifo->in_d);
	deleteGPUmem(fifo->out_d);
	
	return (NULL);
}


void *cpu_consumer (void *q)
{

	struct timeval t1_start,t1_end,t2_start,t2_end;
	double time_d, alltime;

	int i;
	int success=0;
	queue *fifo;	
	fifo = (queue *)q;
	int comp_length=0;
	
	for (i = 0; i < maxiterations; i++) {

		if(exit_signal){
			exit_signal++;
			break;
		}
		gettimeofday(&t1_start,0);	

		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headCS]!=2) {
			pthread_mutex_unlock (fifo->mut);
			pthread_mutex_lock (fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
	
		onestream_finish_GPU(fifo->headCS);

		gettimeofday(&t2_start,0);	

		success=aftercompression_wrapper(fifo->buf[fifo->headCS], blocksize, fifo->bufout[fifo->headCS], &comp_length);
		if(!success){
			printf("After Compression failed. Success %d return size %d\n",success,comp_length);
		}	

		fifo->outsize[fifo->headCS] = comp_length;
		
		gettimeofday(&t2_end,0);
		time_d = (t2_end.tv_sec-t2_start.tv_sec) + (t2_end.tv_usec - t2_start.tv_usec)/1000000.0;
		printf("CPU funccall took:\t%f \t", time_d);

		
		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headCS]++;//=0;
		fifo->headCS++;
		if (fifo->headCS == numblocks)
			fifo->headCS = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->sendready);
		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		printf("CPU whole took:\t%f \n", alltime);
	}
	return (NULL);
}

void *cpu_sender (void *q)
{
	struct timeval t1_start,t1_end,t2_start,t2_end;
	double time_d, alltime;
	FILE *outFile;

	int i, d=0;
	int success=0;
	queue *fifo;	
	fifo = (queue *)q;
	
	for (i = 0; i < maxiterations; i++) 
	{
		if(exit_signal){
			exit_signal++;
			break;
		}
		gettimeofday(&t1_start,0);	

		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headSP]!=3) {
			pthread_cond_wait (fifo->sendready, fifo->mut);
		}
		pthread_mutex_unlock (fifo->mut);
		
		gettimeofday(&t2_start,0);	

		/*WRITE To A FILE TO CHECK CORRECTNESS*/
		if(d==5)
		{
			outFile = fopen("128COMPRESSED.dat", "wb");
			fwrite(fifo->buf[fifo->headSP], fifo->outsize[fifo->headSP], 1, outFile);
			fclose(outFile);
			d++;
		}
		else
			d++;
		
		//
		gettimeofday(&t2_end,0);
		time_d = (t2_end.tv_sec-t2_start.tv_sec) + (t2_end.tv_usec - t2_start.tv_usec)/1000000.0;
		
				
		pthread_mutex_lock (fifo->mut);
		fifo->ledger[fifo->headSP]=0;
		fifo->headSP++;
		if (fifo->headSP == numblocks)
			fifo->headSP = 0;

		pthread_mutex_unlock (fifo->mut);
		
		pthread_cond_signal (fifo->sent);

		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		printf("Sent whole took:\t%f \n", alltime);
		loopnum++;
	}
	return (NULL);
}



queue *queueInit (int maxit,int numb,int bsize)
{
	queue *q;
	maxiterations=maxit;
	numblocks=numb;
	blocksize=bsize;
	
	q = (queue *)malloc (sizeof (queue));
	if (q == NULL) return (NULL);

	int i;
	//alloc bufs
	unsigned char ** buffer;
	unsigned char ** bufferout;
	
	
	/*  allocate storage for an array of pointers */
	buffer = (unsigned char **)malloc((numblocks) * sizeof(unsigned char *));
	if (buffer == NULL) {
		printf("Error: malloc could not allocate buffer\n");
		return;
	}
	bufferout = (unsigned char **)malloc((numblocks) * sizeof(unsigned char *));
	if (bufferout == NULL) {
		printf("Error: malloc could not allocate bufferout\n");
		return;
	}
	  
	/* for each pointer, allocate storage for an array of chars */
	for (i = 0; i < (numblocks); i++) {
		buffer[i] = (unsigned char *)initCPUmem(blocksize * sizeof(unsigned char));
		if (buffer[i] == NULL) {printf ("Memory error, buffer"); exit (2);}
	}
	for (i = 0; i < (numblocks); i++) {
		bufferout[i] = (unsigned char *)initCPUmem(blocksize * 2 * sizeof(unsigned char));
		if (bufferout[i] == NULL) {printf ("Memory error, bufferout"); exit (2);}
	}
	
	q->buf = buffer;
	q->bufout = bufferout;
	
	q->headPG = 0;
	q->headGC = 0;
	q->headCS = 0;
	q->headSP = 0;
	
	q->outsize = (int *)malloc(sizeof(int)*numblocks);
	
	q->ledger = (int *)malloc((numblocks) * sizeof(int));
	if (q->ledger == NULL) {
		printf("Error: malloc could not allocate q->ledger\n");
		return;
	}

	for (i = 0; i < (numblocks); i++) {
		q->ledger[i] = 0;
	}
		
	q->mut = (pthread_mutex_t *) malloc (sizeof (pthread_mutex_t));
	pthread_mutex_init (q->mut, NULL);
	
	q->produced = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->produced, NULL);
	q->compressed = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->compressed, NULL);	
	q->sendready = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->sendready, NULL);	
	q->sent = (pthread_cond_t *) malloc (sizeof (pthread_cond_t));
	pthread_cond_init (q->sent, NULL);
	
	return (q);
}


void signalExitThreads()
{
	exit_signal++;
	while(! (exit_signal > 3));
}


void queueDelete (queue *q)
{
	int i =0;
	
	signalExitThreads();
	
	pthread_mutex_destroy (q->mut);
	free (q->mut);	
	pthread_cond_destroy (q->produced);
	free (q->produced);
	pthread_cond_destroy (q->compressed);
	free (q->compressed);	
	pthread_cond_destroy (q->sendready);
	free (q->sendready);	
	pthread_cond_destroy (q->sent);
	free (q->sent);

	
	for (i = 0; i < (numblocks); i++) {
		deleteCPUmem(q->bufout[i]);	
		deleteCPUmem(q->buf[i]);
	}
	
	deleteGPUStreams();
	
	free(q->buf);
	free(q->bufout);
	free(q->ledger);	
	free(q->outsize);
	free (q);
	
	
	resetGPU();

}


void  init_compression(queue * fifo,int maxit,int numb,int bsize)
{
	maxiterations=maxit;
	numblocks=numb;
	blocksize=bsize;

	printf("Initializing the GPU\n");
	initGPU();
	//create consumer threades
	pthread_create (&congpu, NULL, gpu_consumer, fifo);
	pthread_create (&concpu, NULL, cpu_consumer, fifo);
	pthread_create (&consend, NULL, cpu_sender, fifo);
	
	
	
	return;
}

void join_comp_threads()
{	
	pthread_join (congpu, NULL);
	pthread_join (concpu, NULL);
	pthread_join (consend, NULL);
	exit_signal = 3;
}


