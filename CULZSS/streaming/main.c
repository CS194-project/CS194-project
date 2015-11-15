/***************************************************************************
*          CUDA LZSS test program
*
*   File    : main.c
*   Purpose : Using  CUDA based compression functionality to test compression/decompression 
*   Authors  : Adnan Ozsoy, Martin Swany, Arun Chauhan,Indiana University - Bloomington
*   Update	: December 2012 (last update date)
*   Usage   : 
*           : make
*           :
*           : ./test 
*				-i {number of iterations} 
*				-f {input file} 
*				-b {number of blocks(4 is recommended)} 
*           : Example  ./main -f ESG.data -i 100 -b 4  
*           :    Compressed file is written into 128COMPRESSED.dat 
*			:	 The program for now only accepts input files with the size of power of 2. Testing is done on sizes of 128MB. 

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
	
****************************************************************************/
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include "culzss.h"
#include <sys/time.h>
#include <string.h>
#include <signal.h>
#include "getopt.h"

struct timeval tall_start,tall_end;
double alltime_signal;
int loopcount=0;
queue *fifo;
char * inputfilename;
int maxiters=0;
int numbls=0;
int blsize=0;

// Define the function to be called when ctrl-c (SIGINT) signal is sent to process
void signal_callback_handler(int signum)
{
	gettimeofday(&tall_end,0);
	alltime_signal = (tall_end.tv_sec-tall_start.tv_sec) + (tall_end.tv_usec - tall_start.tv_usec)/1000000.0;
	printf("\tAll the time took:\t%f \n", alltime_signal);
	int sizeinmb = blsize / (1024*1024);
	printf("\tThroughput for %d runs(proc %d)  of %dMB is :\t%lfMbps \n", getloopcount(),loopcount, sizeinmb, (getloopcount()*sizeinmb*8)/alltime_signal);
	
	printf("Caught signal %d\n",signum);
   // Cleanup and close up stuff here
	queueDelete (fifo);
   // Terminate program
   exit(signum);
}

void *producer (void *q)
{
	struct timeval t1_start,t1_end;
	double alltime;
	
    FILE *inFile;//, *outFile, *decFile;  /* input & output files */
	inFile = NULL;
	queue *fifo;
	int i;	


	fifo = (queue *)q;
	
	
	for (i = 0; i < maxiters; i++) {
		
		gettimeofday(&t1_start,0);	
	
		//produce data
		//		
		//read file into memory
		if ((inFile = fopen(inputfilename, "rb")) == NULL){
			printf ("Memory error, temp"); exit (2);
		}	
		
		pthread_mutex_lock (fifo->mut);
		while (fifo->ledger[fifo->headPG]!=0) {
			//printf ("producer: queue FULL.\n");
			pthread_cond_wait (fifo->sent, fifo->mut);
		}
		
		int result = fread (fifo->buf[fifo->headPG],1,blsize,inFile);
		if (result != blsize) {printf ("Reading error1, expected size %d, read size %d ", blsize,result); exit (3);}	

		
		fifo->ledger[fifo->headPG]++;
		fifo->headPG++;
		if (fifo->headPG == numbls)
			fifo->headPG = 0;

		pthread_mutex_unlock (fifo->mut);
		pthread_cond_signal (fifo->produced);
		
		fclose(inFile);
		
		gettimeofday(&t1_end,0);
		alltime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
		printf("Produc took:\t%f \n", alltime);
		

		loopcount++;
		printf("Loop count %d ---------\n",loopcount);
	}
	
	return (NULL);
}

int main (int argc, char* argv[]) 
{
	// Register signal and signal handler
	signal(SIGINT, signal_callback_handler);

	int opt;

	/* parse command line */
	while ((opt = getopt(argc, argv, "f:i:b:s:")) != -1)
    {
      switch(opt)
        {
	
		case 'f':       /* input file name */
		  inputfilename = optarg;
		  break;

		case 'i':       /* number of max itearations */
                maxiters = atoi(optarg);
                break;

		case 'b':       /* number of blocks */
                numbls = atoi(optarg);
                break;
				
		case 's':       /* block size */
                blsize = atoi(optarg);//*1024*1024;
                break;

        }
    }
	
    FILE *filein;//, *outFile, *decFile;  /* input & output files */
	filein = NULL;

	if ((filein = fopen(inputfilename, "rb")) == NULL){
		printf ("Memory error, temp"); exit (2);
	}	
	fseek(filein , 0 , SEEK_END);
	blsize = ftell (filein);
	fclose(filein);
		
	gettimeofday(&tall_start,0);	
	double alltime;
	
	pthread_t pro;
	
	fifo = queueInit (maxiters,numbls,blsize);
	if (fifo ==  NULL) {
		fprintf (stderr, "main: Queue Init failed.\n");
		exit (1);
	}	
	
	//init compression threads
	init_compression(fifo,maxiters,numbls,blsize);

	//create producer
	pthread_create (&pro, NULL, producer, fifo);

	//join all 
	join_comp_threads();
	//join producer
	pthread_join (pro, NULL);
	queueDelete (fifo);

	gettimeofday(&tall_end,0);
	alltime = (tall_end.tv_sec-tall_start.tv_sec) + (tall_end.tv_usec - tall_start.tv_usec)/1000000.0;
	printf("\tAll the time took:\t%f \n", alltime);
	int sizeinmb= blsize / (1024*1024);
	printf("\tThroughput for %d runs of %dMB is :\t%lfMbps \n", maxiters,sizeinmb, (maxiters*sizeinmb*8)/alltime);

	
	//exit
	return 0;

} 



