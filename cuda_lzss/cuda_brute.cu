/***************************************************************************
*          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding
*
*   File    : brute.c
*   Purpose : Implement brute force matching of uncoded strings for LZSS
*             algorithm.
*   Author  : Michael Dipperstein
*   Date    : February 18, 2004
*
****************************************************************************
*
* Brute: Brute force matching routines used by LZSS Encoding/Decoding
*        Routine
* Copyright (C) 2004 - 2007, 2014 by
* Michael Dipperstein (mdipper@alumni.engr.ucsb.edu)
*
* This file is part of the lzss library.
*
* The lzss library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The lzss library is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
* General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
***************************************************************************/

/***************************************************************************
*                             INCLUDED FILES
***************************************************************************/
#include "lzlocal.h"
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <time.h>

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
extern unsigned char slidingWindow[];
extern unsigned char uncodedLookahead[];

static char *window;
static char *lookahead;
static encoded_string_t *cudaReturn;
double debug_time = 0;
double debug_time2 = 0;
/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

/****************************************************************************
*   Function   : InitializeSearchStructures
*   Description: This function initializes structures used to speed up the
*                process of mathcing uncoded strings to strings in the
*                sliding window.  The brute force search doesn't use any
*                special structures, so this function doesn't do anything.
*   Parameters : None
*   Effects    : None
*   Returned   : 0 for success, -1 for failure.  errno will be set in the
*                event of a failure.
****************************************************************************/
int InitializeSearchStructures(void)
{
  cudaMalloc(&window, sizeof(char)*WINDOW_SIZE);
  cudaMalloc(&lookahead, sizeof(char)*MAX_CODED);
  cudaMalloc(&cudaReturn, sizeof(encoded_string_t)*WINDOW_SIZE);
  return 0;
}

__device__ encoded_string_t maxString (encoded_string_t a, encoded_string_t b)
{
    if (a.length > b.length)
        return a;
    else
        return b;
}

struct encoded_binary: public thrust::binary_function<encoded_string_t
                                                      , encoded_string_t
                                                      , encoded_string_t>
{
     __device__ __host__
    encoded_string_t operator() (const encoded_string_t a, const encoded_string_t b)
    {
      if (a.length > b.length)
        return a;
      else
        return b;
    }

};

__global__ void FindMatchKernel (char* window
                                 , char* lookahead
                                 , unsigned int windowHead
                                 , unsigned int uncodedHead
                                 , encoded_string_t *kernelReturn)
{
  //    __shared__ encoded_string_t data[512];
    encoded_string_t matchData;
    unsigned int i;
    unsigned int j;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    i = Wrap(windowHead+idx, WINDOW_SIZE);  /* start at the beginning of the sliding window */
    j = 0;

    while(window[Wrap((i + j), WINDOW_SIZE)] ==
          lookahead[Wrap((uncodedHead + j), MAX_CODED)] && j < MAX_CODED)
    {
        j++;
    }
            
    matchData.length = j;
    matchData.offset = i;
    kernelReturn[idx] = matchData;
}
/****************************************************************************
*   Function   : FindMatch
*   Description: This function will search through the slidingWindow
*                dictionary for the longest sequence matching the MAX_CODED
*                long string stored in uncodedLookahed.
*   Parameters : windowHead - head of sliding window
*                uncodedHead - head of uncoded lookahead buffer
*   Effects    : None
*   Returned   : The sliding window index where the match starts and the
*                length of the match.  If there is no match a length of
*                zero will be returned.
****************************************************************************/
encoded_string_t FindMatch(const unsigned int windowHead,
    unsigned int uncodedHead)
{
  cudaMemcpy(window, slidingWindow, sizeof(char)*WINDOW_SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(lookahead, uncodedLookahead, sizeof(char)*MAX_CODED, cudaMemcpyHostToDevice);
    FindMatchKernel<<<WINDOW_SIZE/1024, 1024>>>(window, lookahead, windowHead, uncodedHead, cudaReturn);
        encoded_string_t result[1];
        cudaMemcpy(result, cudaReturn, sizeof(encoded_string_t), cudaMemcpyDeviceToHost);
        
        /*
    // wrap raw pointer with a device_ptr
   // thrust::device_ptr<encoded_string_t> return_ptr = thrust::device_pointer_cast(cudaReturn);
   // encoded_string_t initial = {0, 0};
   // encoded_string_t result = thrust::reduce(return_ptr, return_ptr + WINDOW_SIZE, initial, encoded_binary());*/
    return result[0];
}

/****************************************************************************
*   Function   : ReplaceChar
*   Description: This function replaces the character stored in
*                slidingWindow[charIndex] with the one specified by
*                replacement.
*   Parameters : charIndex - sliding window index of the character to be
*                            removed from the linked list.
*   Effects    : slidingWindow[charIndex] is replaced by replacement.
*   Returned   : 0 for success, -1 for failure.  errno will be set in the
*                event of a failure.
****************************************************************************/
int ReplaceChar(const unsigned int charIndex, const unsigned char replacement)
{
    slidingWindow[charIndex] = replacement;
    return 0;
}

void end ()
{
    cudaFree(window);
  cudaFree(lookahead);
  cudaFree(cudaReturn);
  cudaDeviceReset();
}
