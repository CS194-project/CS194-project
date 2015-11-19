/*
 * CUDA LZSS
 * Author: Harry He
 * Date: November 10, 2015
 * This code is campatible with Dippersteain's LZSS decoder.
 * Only compression is implemented.
 * Uses some of Dipperstein's code.
 * The following is his licence.
 * /


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
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <errno.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <assert.h>
#include <cuda.h>
#include <limits.h>

#include "bitfile.h"
#include "optlist.h"
/***************************************************************************
 *                            GLOBAL VARIABLES
 ***************************************************************************/

typedef struct
{
  unsigned short offset;	/* offset to start of longest match */
  unsigned short length;	/* length of longest match */
} encoded_string_t;


const int WINDOW_BITS = 12;
const int WINDOW_SIZE = (1 << WINDOW_BITS);

const int HASH_SHIFT = 5;
const int HASH_BITS = 13;	/* HASH_SHIFT * MIN_MATCH >= HASH_BITS. */
const int HASH_SIZE = (1 << HASH_BITS);
const int HASH_MASK = (HASH_SIZE - 1);

const int LENGTH_BITS = 2;
const int MIN_MATCH = 3;
const int MAX_MATCH = (1 << LENGTH_BITS) + MIN_MATCH - 1;	/* 6  */

const int MAX_PROCESS_SIZE = (16 * 1024 * 1024);	/* Process 16M at one one. */
const int EXTRA_BUF = (2 * WINDOW_SIZE);	/* Extra BUF to avoid memory out of
                                             * bound. */

const int UNCODED = 1;
const int ENCODED = 0;

const int CUDA_BLOCK_SIZE = (1 * 1024 * 1024);	/* Size of bytes processed per kernel
                                                   launch. */
const int CUDA_NUM_BLOCKS = 1;	/* Max 4 blocks in GT740 if we run 1024 threads
                                   in each block. One kernel only runs in block
                                   in order to overlap kernel copy and execution. */

/* Number of streams. One stream must have a one-to-one relationship to a
   kernel instance. */
const int CUDA_NUM_STREAMS = (MAX_PROCESS_SIZE / CUDA_BLOCK_SIZE) + 1;

/* wraps array index within array bounds (assumes value < 2 * limit) */
#define Wrap(value, limit)                              \
  (((value) < (limit)) ? (value) : ((value) - (limit)))

/***************************************************************************/
/* Global Variables. */
unsigned char *host_in;
unsigned char *device_in;
encoded_string_t *host_encode;
encoded_string_t *device_encode;
cudaStream_t streams[CUDA_NUM_STREAMS];

/***************************************************************************
 *                               PROTOTYPES
 ***************************************************************************/
void
checkCPUError (const char *msg)
{
  const int BUF_SIZE = 1024;
  char buf[BUF_SIZE];
  if (errno != 0 && errno != EEXIST)
    {
      char *errmsg = strerror_r (errno, buf, 1024);
      printf ("CPU error: %s: %s.\n", msg, errmsg);
      exit (1);
    }
}

void
checkCudaError (const char *msg)
{
  cudaError_t err = cudaGetLastError ();
  if (cudaSuccess != err)
    {
      fprintf (stderr, "Cuda error: %s: %s.\n", msg,
               cudaGetErrorString (err));
      exit (EXIT_FAILURE);
    }
}

__global__ void
lzss_kernel (const unsigned char *__restrict__ in_g,
             encoded_string_t * __restrict__ encode, int grid_size,
             int is_firstblock)
{
  __shared__ unsigned char in[WINDOW_SIZE * 2];	/* Note that WINDOW_SIZE must be a
                                           multiple of blockDimension(1024).
                                           First half are window and second half are
                                           lookahead. */
  __shared__ unsigned short hashtable[HASH_SIZE];
  const int STEP_SIZE = 64 * 1024;
  /* We do 64KB every step. (Actual 64KB - 4KB (window size) because of
     overlapping).
     This is because we use unsigned short as the
     type of hashtable to maximize the number of indexes in hashtable.
     It can store the indexes to the maximum to 65535. */

  while (grid_size > 0)
    {
      int block_size = grid_size - (STEP_SIZE * blockIdx.x -
                                    WINDOW_SIZE * blockIdx.x);
      if (block_size > STEP_SIZE)
        block_size = STEP_SIZE;
      encode += (STEP_SIZE - WINDOW_SIZE) * blockIdx.x;
      in_g += (STEP_SIZE - WINDOW_SIZE) * blockIdx.x;

      /* initialize shared memory. */
      /* Note that in_g needs to have some extra space at the end to avoid invalid
         memory access */
      for (int i = threadIdx.x * 4; i < WINDOW_SIZE * 2; i += blockDim.x * 4)
        {
          *((int *) (in + i)) = *((int *) (in_g + i));
        }
      __syncthreads ();

      /* Compute hash of initial sliding window. */
      for (int ii = threadIdx.x * 4; ii < WINDOW_SIZE; ii += blockDim.x * 4)
        {
          for (int i = ii; i < ii + 4; i++)
            {
              int hash = 0;
              hash = (((hash) << HASH_SHIFT) ^ (in[i]));
              hash = (((hash) << HASH_SHIFT) ^ (in[i + 1]));
              hash = (((hash) << HASH_SHIFT) ^ (in[i + 2]));
              hash = hash & HASH_MASK;
              hashtable[hash] = i;
            }
        }
      __syncthreads ();

      int hash0 = 0, index0 = threadIdx.x;
      int hash1 = 0, index1 = threadIdx.x;
      int hash2 = 0, index2 = threadIdx.x;
      int hash3 = 0, index3 = threadIdx.x;
      for (int uncodedHead = WINDOW_SIZE; uncodedHead < block_size;
           uncodedHead += WINDOW_SIZE)
        {
          int end = WINDOW_SIZE + min (block_size - uncodedHead, WINDOW_SIZE);

          for (int i = threadIdx.x + WINDOW_SIZE; i < 2 * WINDOW_SIZE;
               i += blockDim.x)
            {
              unsigned int hash = 0;
              unsigned char char0 = in[i];
              unsigned char char1 = 0;
              unsigned char char2 = 0;
              unsigned char char3 = 0;
              unsigned char char4 = 0;
              unsigned char char5 = 0;
              encoded_string_t match_data;
              int match_length;
              unsigned int prev;
              match_data.length = 0;
              match_data.offset = 0;

              /* Load characters */
              if (i < end - MAX_MATCH + 1)
                {
                  char0 = in[i];
                  char1 = in[i + 1];
                  char2 = in[i + 2];
                  char3 = in[i + 3];
                  char4 = in[i + 4];
                  char5 = in[i + 5];
                  /* Compute hash key of 3 characters. */
                  hash = (((hash) << HASH_SHIFT) ^ (char0));
                  hash = (((hash) << HASH_SHIFT) ^ (char1));
                  hash = (((hash) << HASH_SHIFT) ^ (char2));
                  hash = hash & HASH_MASK;
                }

              /* Check most recent hash. */
              __syncthreads ();
              prev = hashtable[hash];
              prev = prev + WINDOW_SIZE - uncodedHead;
              match_length = 0;

              if (prev > 0
                  && prev < i - MAX_MATCH + 1
                  && i - prev <= WINDOW_SIZE && i < end - MAX_MATCH + 1)
                {
                  int temp = 1;	/* Used to check match. */
                  temp *= (in[prev] == char0);
                  match_length += temp;
                  temp *= (in[prev + 1] == char1);
                  match_length += temp;
                  temp *= (in[prev + 2] == char2);
                  match_length += temp;
                  temp *= (in[prev + 3] == char3);
                  match_length += temp;
                  temp *= (in[prev + 4] == char4);
                  match_length += temp;
                  temp *= (in[prev + 5] == char5);
                  match_length += temp;
                }
              if (match_length > match_data.length)
                {
                  match_data.offset = i - prev;
                  match_data.length = match_length;
                }

              /* Check second recent hash. */
              __syncthreads ();
              hashtable[hash1] = index1;
              __syncthreads ();

              prev = hashtable[hash];
              prev = prev + WINDOW_SIZE - uncodedHead;
              match_length = 0;
              if (prev > 0
                  && prev < i - MAX_MATCH + 1
                  && i - prev <= WINDOW_SIZE && i < end - MAX_MATCH + 1)
                {
                  int temp = 1;	/* Used to check match. */
                  temp *= (in[prev] == char0);
                  match_length += temp;
                  temp *= (in[prev + 1] == char1);
                  match_length += temp;
                  temp *= (in[prev + 2] == char2);
                  match_length += temp;
                  temp *= (in[prev + 3] == char3);
                  match_length += temp;
                  temp *= (in[prev + 4] == char4);
                  match_length += temp;
                  temp *= (in[prev + 5] == char5);
                  match_length += temp;
                }
              if (match_length > match_data.length)
                {
                  match_data.offset = i - prev;
                  match_data.length = match_length;
                }


              /* Check third recent hash. */
              __syncthreads ();
              hashtable[hash2] = index2;
              __syncthreads ();

              prev = hashtable[hash];
              prev = prev + WINDOW_SIZE - uncodedHead;
              match_length = 0;
              if (prev > 0
                  && prev < i - MAX_MATCH + 1
                  && i - prev <= WINDOW_SIZE && i < end - MAX_MATCH + 1)
                {
                  int temp = 1;	/* Used to check match. */
                  temp *= (in[prev] == char0);
                  match_length += temp;
                  temp *= (in[prev + 1] == char1);
                  match_length += temp;
                  temp *= (in[prev + 2] == char2);
                  match_length += temp;
                  temp *= (in[prev + 3] == char3);
                  match_length += temp;
                  temp *= (in[prev + 4] == char4);
                  match_length += temp;
                  temp *= (in[prev + 5] == char5);
                  match_length += temp;
                }
              if (match_length > match_data.length)
                {
                  match_data.offset = i - prev;
                  match_data.length = match_length;
                }

              /* Check forth recent hash. */
              __syncthreads ();
              hashtable[hash3] = index3;
              __syncthreads ();

              prev = hashtable[hash];
              prev = prev + WINDOW_SIZE - uncodedHead;
              match_length = 0;
              if (prev > 0
                  && prev < i - MAX_MATCH + 1 && i - prev <= WINDOW_SIZE
                  && i < end - MAX_MATCH + 1)
                {
                  int temp = 1;	/* Used to check match. */
                  temp *= (in[prev] == char0);
                  match_length += temp;
                  temp *= (in[prev + 1] == char1);
                  match_length += temp;
                  temp *= (in[prev + 2] == char2);
                  match_length += temp;
                  temp *= (in[prev + 3] == char3);
                  match_length += temp;
                  temp *= (in[prev + 4] == char4);
                  match_length += temp;
                  temp *= (in[prev + 5] == char5);
                  match_length += temp;
                }
              if (match_length > match_data.length)
                {
                  match_data.offset = i - prev;
                  match_data.length = match_length;
                }

              /* Update recent hash */
              hash3 = hash2;
              index3 = index2;
              hash2 = hash1;
              index2 = index1;
              hash1 = hash0;
              index1 = index0;
              hash0 = hash0;
              index0 = i + uncodedHead - WINDOW_SIZE;

              /* Check current hash. */
              __syncthreads ();
              hashtable[hash] = index0;
              __syncthreads ();

              prev = hashtable[hash];
              prev = prev + WINDOW_SIZE - uncodedHead;
              match_length = 0;
              if (prev > 0
                  && prev < i - MAX_MATCH + 1 && i - prev <= WINDOW_SIZE
                  && i < end - MAX_MATCH + 1)
                {
                  int temp = 1;	/* Used to check match. */
                  temp *= (in[prev] == char0);
                  match_length += temp;
                  temp *= (in[prev + 1] == char1);
                  match_length += temp;
                  temp *= (in[prev + 2] == char2);
                  match_length += temp;
                  temp *= (in[prev + 3] == char3);
                  match_length += temp;
                  temp *= (in[prev + 4] == char4);
                  match_length += temp;
                  temp *= (in[prev + 5] == char5);
                  match_length += temp;
                }

              if (match_length > match_data.length)
                {
                  match_data.offset = i - prev;
                  match_data.length = match_length;
                }

              /* Don't compress first block because initial window can be arbitrary. */
              if ((blockIdx.x == 0 && uncodedHead == WINDOW_SIZE
                  && is_firstblock) || match_data.length < MIN_MATCH)
                {
                  match_data.offset = 0;
                  match_data.length = (unsigned short)char0; /* Store string literal in
                match_data.length. (We know its a literal by match_data.offset
                == 0) */
                }
              /* Done at this position. Store to global memory. */
              if (i < end)
                {
                  encode[i + uncodedHead - WINDOW_SIZE] = match_data;
                }
            }

          __syncthreads ();

          /* Move sliding window. */
          for (int j = threadIdx.x * 4; j < WINDOW_SIZE; j += blockDim.x * 4)
            {
              *((int *) (in + j)) = *((int *) (in + j + WINDOW_SIZE));
            }

          for (int j = threadIdx.x * 4; j < WINDOW_SIZE; j += blockDim.x * 4)
            {
              *((int *) (in + WINDOW_SIZE + j)) =
                *((int *) (in_g + uncodedHead + WINDOW_SIZE + j));
            }

          __syncthreads ();
        }

      grid_size -= STEP_SIZE - WINDOW_SIZE;
      encode += (STEP_SIZE - WINDOW_SIZE) * gridDim.x;
      in_g += (STEP_SIZE - WINDOW_SIZE) * gridDim.x;
    }
}

void
culzss_init ()
{

  for (int i = 0; i < CUDA_NUM_STREAMS; i++)
    cudaStreamCreate (&streams[i]);
  checkCudaError ("Cuda Create stream.");

  /* Need initial WINDOW_SIZE bytes as initial window. EXTRA BUF to avoid out
   * of bound memory error. */
  cudaHostAlloc (&host_in,
                 sizeof (*host_in) * (MAX_PROCESS_SIZE + WINDOW_SIZE +
                                     EXTRA_BUF), cudaHostAllocDefault);
  checkCudaError ("Allocate host_in");
  cudaMalloc (&device_in,
              sizeof (*device_in) * (MAX_PROCESS_SIZE + WINDOW_SIZE +
                                     EXTRA_BUF));
  checkCudaError ("Allocate device_in");

  cudaHostAlloc (&host_encode,
                 sizeof (*host_encode) * (MAX_PROCESS_SIZE + WINDOW_SIZE +
                                          EXTRA_BUF), cudaHostAllocDefault);
  checkCudaError ("Allocate host_encode");
  cudaMalloc (&device_encode,
              sizeof (*device_encode) * (MAX_PROCESS_SIZE + WINDOW_SIZE +
                                         EXTRA_BUF));
  checkCudaError ("Allocate device_in");
}

void
culzss_destroy ()
{
  /* Clean up */
  for (int i = 0; i < CUDA_NUM_STREAMS; i++)
    cudaStreamDestroy (streams[i]);

  cudaFree (device_in);
  cudaFree (device_encode);

  cudaFreeHost (host_in);
  cudaFreeHost (host_encode);
}

int
do_work (FILE * fpIn, FILE * fpOut)
{
  /* Input */
  fseek (fpIn, 0L, SEEK_END);
  checkCPUError ("Input seek.");
  const int total_size = ftell (fpIn);
  checkCPUError ("Input tell.");
  rewind (fpIn);
  checkCPUError ("Input rewind.");

  bit_file_t *bfpOut = MakeBitFile (fpOut, BF_WRITE);
  if (NULL == bfpOut)
    {
      perror ("Making Output File a BitFile");
      return -1;
    }
  /* CUDA initialize */

  culzss_init ();

  /* Initial buffer */

  float total_compute_milliseconds = 0;
  float total_output_milliseconds = 0;
  int total_remaining = total_size;
  while (total_remaining > 0)
    {
      int processing_size = std::min (total_remaining, MAX_PROCESS_SIZE);
      total_remaining -= processing_size;

      fread (host_in + WINDOW_SIZE, 1, processing_size, fpIn);
      checkCPUError ("read input file");

      cudaEvent_t gpu_start, cpu_start,  stop;
      cudaEventCreate (&gpu_start);
      cudaEventCreate (&cpu_start);
      cudaEventCreate (&stop);

      cudaEventRecord (gpu_start);

      /* Don't compress first block */
      int is_firstblock = 1;

      /* Copy data into GPU */
      for (int i = 0, block = 0, stream = 0;
           i < processing_size + WINDOW_SIZE;
           i += (CUDA_BLOCK_SIZE - WINDOW_SIZE) * CUDA_NUM_BLOCKS, block =
             (block + 1) % CUDA_NUM_BLOCKS, stream =
             (stream + 1) % CUDA_NUM_STREAMS)
        {
          /* Don't copy initial window, which has been copied by previous kernel. */
          /* Note: Not sure if the ordering is correct if GPU has two copy engines. */
          int size =
            std::min (processing_size + WINDOW_SIZE - i,
                      CUDA_BLOCK_SIZE * CUDA_NUM_BLOCKS - (CUDA_NUM_BLOCKS -
                                                           1) * WINDOW_SIZE);
          cudaMemcpyAsync (device_in + i + WINDOW_SIZE,
                           host_in + i + WINDOW_SIZE,
                           sizeof (*host_in) * max (size - WINDOW_SIZE, 0),
                           cudaMemcpyHostToDevice, streams[stream]);
          checkCudaError ("copy from host_in to device_in");
        }

      /* Call kernel */
      for (int i = 0, block = 0, stream = 0;
           i < processing_size + WINDOW_SIZE;
           i += (CUDA_BLOCK_SIZE - WINDOW_SIZE) * CUDA_NUM_BLOCKS, block =
             (block + 1) % CUDA_NUM_BLOCKS, stream =
             (stream + 1) % CUDA_NUM_STREAMS)
        {
          int size =
            std::min (processing_size + WINDOW_SIZE - i,
                      CUDA_BLOCK_SIZE * CUDA_NUM_BLOCKS - (CUDA_NUM_BLOCKS -
                                                           1) * WINDOW_SIZE);
          lzss_kernel <<< CUDA_NUM_BLOCKS, 1024, 0,
            streams[stream] >>> (device_in + i, device_encode + i, size,
                                 is_firstblock);
          checkCudaError ("launch lzss_kernel.");
          is_firstblock = 0;
        }

      /* Copy result to CPU. */
      for (int i = 0, block = 0, stream = 0;
           i < processing_size + WINDOW_SIZE;
           i += (CUDA_BLOCK_SIZE - WINDOW_SIZE) * CUDA_NUM_BLOCKS, block =
             (block + 1) % CUDA_NUM_BLOCKS, stream =
             (stream + 1) % CUDA_NUM_STREAMS)
        {
          /* Don't copy initial window, which has been copied by previous kernel. */
          /* Note: Not sure if the ordering is correct if GPU has two copy engines. */
          int size =
            std::min (processing_size + WINDOW_SIZE - i,
                      CUDA_BLOCK_SIZE * CUDA_NUM_BLOCKS - (CUDA_NUM_BLOCKS -
                                                           1) * WINDOW_SIZE);

          cudaMemcpyAsync (host_encode + i + WINDOW_SIZE,
                           device_encode + i + WINDOW_SIZE,
                           sizeof (*device_encode) * std::max (size -
                                                               WINDOW_SIZE,
                                                               0),
                           cudaMemcpyDeviceToHost, streams[stream]);
          checkCudaError ("Copy from device_on to host_in.");
        }

      checkCudaError ("Copy from device_encode to host_encode. ");
      cudaEventRecord (stop);
      // int gpu_done = 0;

      /* Record CPU output time. */
      cudaEventRecord (cpu_start);

      /* Output results */
      int dipperstein_lzss_uncodedHead = 0;	/* To convert to Dipperstein lzss
                                             * format. */
      encoded_string_t matchData;

      /* CPU and GPU overlapping */
      for (int i = 0, stream = 0; i < processing_size + WINDOW_SIZE;
           i += (CUDA_BLOCK_SIZE - WINDOW_SIZE) * CUDA_NUM_BLOCKS
             , stream = (stream + 1) % CUDA_NUM_STREAMS)
        {
          int size = std::min (processing_size + WINDOW_SIZE - i,
                      CUDA_BLOCK_SIZE * CUDA_NUM_BLOCKS - (CUDA_NUM_BLOCKS -
                                                           1) * WINDOW_SIZE);
          cudaStreamSynchronize(streams[stream]);
          for (int j = i+WINDOW_SIZE; j < i+size;
               j += (matchData.offset==0?1:matchData.length))
            {
              /*if (!gpu_done && cudaEventQuery(stop) == cudaSuccess)
                {
                  float milliseconds;
                  gpu_done = 1;
                  cudaEventElapsedTime (&milliseconds, gpu_start, stop);
                  total_compute_milliseconds += milliseconds;
                  }*/
              matchData = host_encode[j];

              if (matchData.offset == 0) /* In this case, matchData.length
                                            contains the character literal. */
                {
                  /* not long enough match.  write uncoded flag and character */
                  BitFilePutBit (UNCODED, bfpOut);
                  BitFilePutChar (((char)matchData.length), bfpOut);
                }
              else
                {
                  /* To convert to Dippersion lzss format. */
                  unsigned short offset = Wrap (dipperstein_lzss_uncodedHead
                                           + WINDOW_SIZE -
                                           matchData.offset, WINDOW_SIZE);

                  unsigned int adjustedLen;

                  /* adjust the length of the match so minimun encoded len is 0 */
                  adjustedLen = matchData.length - MIN_MATCH;

                  /* match length > MAX_UNCODED.  Encode as offset and length. */
                  BitFilePutBit (ENCODED, bfpOut);
                  BitFilePutBitsNum (bfpOut, &offset, WINDOW_BITS,
                                     sizeof (unsigned short));
                  BitFilePutBitsNum (bfpOut, &adjustedLen, LENGTH_BITS,
                                     sizeof (unsigned short));
                }
              dipperstein_lzss_uncodedHead = Wrap (dipperstein_lzss_uncodedHead
                                                   + (matchData.offset==0?1:matchData.length),
                                                   WINDOW_SIZE);
            }
        }
      float milliseconds;
      cudaEventRecord (stop);
      cudaEventSynchronize (stop);
      cudaEventElapsedTime (&milliseconds, cpu_start, stop);
      total_output_milliseconds += milliseconds;

      cudaEventDestroy (gpu_start);
      cudaEventDestroy (cpu_start);
      cudaEventDestroy (stop);
    }

  BitFileToFILE (bfpOut);
  printf ("No IO time: %f ms, No IO speed: %f MB/s\n",
          total_compute_milliseconds,
          total_size / total_compute_milliseconds / 1e3);
  printf ("Output time: %f ms, No IO speed: %f MB/s\n",
          total_output_milliseconds,
          total_size / total_output_milliseconds / 1e3);

  culzss_destroy ();
  return 0;
}



int
main (int argc, char *argv[])
{
  errno = 0;
  FILE *fpIn = NULL;
  FILE *fpOut = NULL;
  option_t *optList;
  option_t *thisOpt;

  /* parse command line */
  optList = GetOptList (argc, argv, (const char *) "i:o:h?");
  thisOpt = optList;

  while (thisOpt != NULL)
    {
      switch (thisOpt->option)
        {
        case 'i':		/* input file name */
          if (fpIn != NULL)
            {
              fprintf (stderr, "Multiple input files not allowed.\n");
              fclose (fpIn);

              if (fpOut != NULL)
                {
                  fclose (fpOut);
                }

              FreeOptList (optList);
              return -1;
            }

          /* open input file as binary */
          fpIn = fopen (thisOpt->argument, "rb");
          if (fpIn == NULL)
            {
              perror ("Opening input file");

              if (fpOut != NULL)
                {
                  fclose (fpOut);
                }

              FreeOptList (optList);
              return -1;
            }
          break;

        case 'o':		/* output file name */
          if (fpOut != NULL)
            {
              fprintf (stderr, "Multiple output files not allowed.\n");
              fclose (fpOut);

              if (fpIn != NULL)
                {
                  fclose (fpIn);
                }

              FreeOptList (optList);
              return -1;
            }

          /* open output file as binary */
          fpOut = fopen (thisOpt->argument, "wb");
          if (fpOut == NULL)
            {
              perror ("Opening output file");

              if (fpIn != NULL)
                {
                  fclose (fpIn);
                }

              FreeOptList (optList);
              return -1;
            }
          break;

        case 'h':
        case '?':
          printf ("Usage: %s <options>\n\n", FindFileName (argv[0]));
          printf ("options:\n");
          printf ("  -i <filename> : Name of input file.\n");
          printf ("  -o <filename> : Name of output file.\n");
          printf ("  -h | ?  : Print out command line options.\n\n");
          printf ("Default: %s -c -i stdin -o stdout\n",
                  FindFileName (argv[0]));

          FreeOptList (optList);
          return 0;
        }
      optList = thisOpt->next;
      free (thisOpt);
      thisOpt = optList;
    }
  /* use stdin/out if no files are provided */
  if (fpIn == NULL)
    {
      fpIn = stdin;
    }

  if (fpOut == NULL)
    {
      fpOut = stdout;
    }


  do_work (fpIn, fpOut);

  return 0;
}
