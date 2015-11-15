/***************************************************************************
*          CUDA LZSS test program
*
*   File    : test.c
*   Purpose : Using  cudalzss library to test compression/decompression 
*   Authors  : Adnan Ozsoy, Martin Swany, Indiana University - Bloomington
*   Date    : April 11, 2011
*   Usage   : 
*           : ./test -i {inputfile} -o {outputfile} -x {numofthreads(128 is recommended)} -d (for decompression) 
*           : Example  ./test -i ../benchmarkdata/esg/ESG.data -o DecompressedESG.data -x 128 -d
*           :    Compressed file is written into DecompressedESG.data 
*           :    Since the -d option is on, the test also decompressed the compressed file to check correctness and  
*                    it is all written to hard coded file named "01outDEC"

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

 

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "getopt.h"


int  gpu_compress(unsigned char * buffer, long buf_length,unsigned char * compressed_buffer, long * comp_length, int compression_type, int wsize, int numthre);
int  gpu_decompress(unsigned char * buffer, long buf_length,unsigned char * decompressed_buffer, long * comp_length, int compression_type, int wsize, int numthre);
 
extern unsigned char * decompressed_buffer; // memory string of buffer

int main (int argc, char* argv[]){
	struct timeval t1_start,t1_end,total_start,total_end;
	double alltime, gputime, iotime;
	
	gettimeofday(&total_start,0);

  int opt, wsize, numthre, decnumthre, ori_size;
  FILE *inFile, *outFile, *decFile;  /* input & output files */
  char * inf; 
  char * outf;
  int decompress =0;
  int success=0;
  /* initialize data */
  inFile = NULL;
  outFile = NULL;
	 
	
  /* parse command line */
  while ((opt = getopt(argc, argv, "i:x:d:o:y:")) != -1)
    {
      switch(opt)
        {
	
		case 'i':       /* input file name */
		  if (inFile != NULL)
			{
			  fprintf(stderr, "Multiple input files not allowed.\n");
			  fclose(inFile);

			  if (outFile != NULL)
			{
			  fclose(outFile);
			}

			  exit(EXIT_FAILURE);
			}
		  else if ((inFile = fopen(optarg, "rb")) == NULL)
			{
			  perror("Opening inFile");

			  if (outFile != NULL)
			{
			  fclose(outFile);
			}

			  exit(EXIT_FAILURE);
			}
			inf = optarg;
		  break;

		case 'x':       /* number of threads */
                numthre = atoi(optarg);
                break;

		case 'y':       /* number of threads */
                decnumthre = atoi(optarg);
                break;

		case 'w':       /* window size */
                wsize = atoi(optarg);
                break;

		case 'd':       
		  decompress = 1;
		  if ((decFile = fopen(optarg, "wb")) == NULL)
			{
			  printf("\ndecFile %s\n", optarg);
			  perror("Opening decFile");
			}
			outf = optarg;
                  break;


		case 'o':       /* output file name */
		  if (outFile != NULL)
			{
			  fprintf(stderr, "Multiple output files not allowed.\n");
			  fclose(outFile);

			  if (inFile != NULL)
			{
			  fclose(inFile);
			}

			  exit(EXIT_FAILURE);
			}
		  else if ((outFile = fopen(optarg, "wb")) == NULL)
			{
			  perror("Opening outFile");

			  if (outFile != NULL)
			{
			  fclose(inFile);
			}

			  exit(EXIT_FAILURE);
			}
		  break;

	
        }
    }

	
  /* validate command line */
  if (inFile == NULL)
    {
      fprintf(stderr, "Input file must be provided\n");
      fprintf(stderr, "Enter \"lzss -?\" for help.\n");

      if (outFile != NULL)
        {
	  fclose(outFile);
        }

      exit (EXIT_FAILURE);
    }
  else if (outFile == NULL)
    {
      fprintf(stderr, "Output file must be provided\n");
      fprintf(stderr, "Enter \"lzss -?\" for help.\n");

      if (inFile != NULL)
        {
	  fclose(inFile);
        }

      exit (EXIT_FAILURE);
    }
	
  //printf("0 %d ",numthre);

  unsigned char *buffer; // memory buffer
 
  
  fseek (inFile , 0 , SEEK_END);
  long buf_length = ftell (inFile);
  long comp_length;
  rewind (inFile);


  // allocate memory to contain the whole file:
  buffer = (unsigned char*) malloc (sizeof(char)*buf_length);
  if (buffer == NULL) {printf ("Memory error"); exit (2);}

  wsize = (int)buf_length;
  
  //read file into memory
  int result = fread (buffer,1,buf_length,inFile);
  if (result != buf_length) {printf ("Reading error1 "); exit (3);}

  comp_length=0;

  //reset gpu time
  gettimeofday(&t1_start,0);
  
  //send file to gpu for compression
  success=gpu_compress(buffer, buf_length, buffer, &comp_length, 0, wsize, numthre);
  if(!success){
	printf("Compression failed. Success %d\n",success);
	}
  else{
  	printf("Compression Success %d length %ld \n",success,comp_length);
	gettimeofday(&t1_end,0);
	gputime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;
	printf("Compression took: \n\tOn memory  %lf secs \n", gputime);

	//reset IO time
	//start = clock();
   	gettimeofday(&t1_start,0);		  
	
	int i=0;
	fwrite(buffer, comp_length, 1, outFile);
	
	gettimeofday(&t1_end,0);
	iotime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;	  
	printf("\tIO took: %f secs",iotime);
	printf("\n\tTotal: %f secs",iotime+gputime);

	  if(decompress){
		
		buf_length=comp_length;
		comp_length=0;

		gettimeofday(&t1_start,0);	

		printf("\nIn decompression\n");
		//send file to gpu
		gpu_decompress(buffer, buf_length, decompressed_buffer, &comp_length, 0, wsize, decnumthre);

		gettimeofday(&t1_end,0);
		gputime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;

		printf("Decompresion took: \n\tOn memory %f secs\n", gputime);

		//reset IO time
		gettimeofday(&t1_start,0);	

	    fwrite(decompressed_buffer, comp_length, 1, decFile);

		printf("Decompression  length %ld \n",comp_length);

		free(decompressed_buffer);

	  }
	  
	  if(decompress){
		gettimeofday(&t1_end,0);
		iotime = (t1_end.tv_sec-t1_start.tv_sec) + (t1_end.tv_usec - t1_start.tv_usec)/1000000.0;	  
		printf("\tIO: %f secs\n\tTotal %f secs",iotime,iotime+gputime);
	  }
	  printf("\n");

	}
  //clean memory
  free(buffer);
  fclose(inFile);
  fclose(outFile);

  // check difference
  if(decompress){
	fclose(decFile);

  if(success){
		char buf[256];
		sprintf(buf, "diff %s %s > error.log", inf, outf);
		printf("%s\n", buf);
		if(system(buf)>0){
			printf("DO NOT MATCH!!! See error.log\n");
		}
		printf("------ \n head error.log\n------\n");
		system("head error.log");
	}
  }
 
  return 0;

}

