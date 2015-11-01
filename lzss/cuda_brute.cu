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

/***************************************************************************
*                            GLOBAL VARIABLES
***************************************************************************/
/* cyclic buffer sliding window of already read characters */
extern unsigned char slidingWindow[];
extern unsigned char uncodedLookahead[];

/* cuda device buffer.
 * TODO: missing cudaFree() */
static unsigned char *cuda_window;
static unsigned char *cuda_lookahead;

/***************************************************************************
*                                FUNCTIONS
***************************************************************************/

/****************************************************************************
*   Function   : InitializeSearchStructures
*   Description: This function initializes structures used to speed up the
*                process of mathcing uncoded strings to strings in the
*                sliding window.
*   Parameters : None
*   Effects    : Initialize cuda_window and cuda_lookahead.
*   Returned   : 0 for success, -1 for failure.  errno will be set in the
*                event of a failure.
****************************************************************************/
int InitializeSearchStructures(void)
{
    cudaMalloc(&cuda_window, sizeof(char)*WINDOW_SIZE);
    cudaMalloc(&cuda_lookahead, sizeof(char)*MAX_CODED);
    cudaMemcpy(cuda_window, slidingWindow, sizeof(char)*WINDOW_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_lookahead, uncodedLookahead, sizeof(char)*WINDOW_SIZE, cudaMemcpyHostToDevice);
    return 0;
}

__device__ encoded_string_t maxString (encoded_string_t a, encoded_string_t b
                                       , unsigned int windowHead)
{
    if (a.length > b.length)
        return a;
    else if (a.length < b.length)
        return b;
    else
    {
        int al = a.offset-windowHead;
        if (al < 0)
            al = WINDOW_SIZE - windowHead - al;
        int bl = b.offset-windowHead;
        if (bl < 0)
            bl = WINDOW_SIZE - windowHead - bl;
        if (al < bl)
            return a;
        else
            return b;
    }
}

__global__ void FindMatchKernel (unsigned char* window
                                 , unsigned char* lookahead
                                 , unsigned int windowHead
                                 , unsigned int uncodedHead
                                 , encoded_string_t *kernelReturn)
{
    __shared__ encoded_string_t data[512];
    encoded_string_t matchData;
    unsigned int i;
    unsigned int j;

    matchData.length = 0;
    matchData.offset = 0;
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    i = Wrap(windowHead+idx, WINDOW_SIZE);  /* start at the beginning of the
                                             * sliding window */
    j = 0;

    while (1)
    {
        if (window[i] == lookahead[uncodedHead])
        {
            /* we matched one. how many more match? */
            j = 1;

            while(window[Wrap((i + j), WINDOW_SIZE)] ==
                lookahead[Wrap((uncodedHead + j), MAX_CODED)])
            {
                if (j >= MAX_CODED)
                {
                    break;
                }
                j++;
            }

            if (j > matchData.length)
            {
                matchData.length = j;
                matchData.offset = i;
            }
        }

        if (j >= MAX_CODED)
        {
            matchData.length = MAX_CODED;
            break;
        }
    }

    int id = threadIdx.x;
    data[id] = matchData;
    __syncthreads();
    for (int i = 256; i > 0; i/=2)
    {
        if (id < i)
            data[id] = maxString(data[id], data[id+i], windowHead);
        __syncthreads ();
    }

    if (id < 1)
        kernelReturn[0] = data[0];
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
    encoded_string_t *cudaReturn;
    encoded_string_t hostReturn[1];
    cudaMalloc(&cudaReturn, sizeof(encoded_string_t));
    FindMatchKernel<<<WINDOW_SIZE/512, 512>>>(cuda_window, cuda_lookahead
                                          , windowHead, uncodedHead, cudaReturn);
    cudaMemcpy(hostReturn, cudaReturn, sizeof(encoded_string_t)
               , cudaMemcpyDeviceToHost);
    cudaFree(cudaReturn);
    return hostReturn[0];
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
    /* UNUSED */
    slidingWindow[charIndex] = replacement;
    return 0;
}

/****************************************************************************
*   TODO: Make the description more clearly.
*
*   Function   : UpdateWindowAndLookAhead
*   Description: This function reads newDataLength's characters from newData
*                , and push them into lookahead.
*                In order to keep lookahead's size constant, pop
*                the oldeest stuff from lookahead and push it into sliding
*                window. Pop oldest stuff in sliding window to keep its Size
*                constant.
*
*                In this cuda case, sliding window is cuda array "cuda_window"
*                and lookahead is cuda array "cuda_lookahead".
*
*                Sliding Window and lookahead are stored in circular loop.
*
*                You can consider cuda_window[windowHead] is the beginning of
*                window and cuda_lookahead[uncodedHead] is the starting of
*                lookahead.
*                We pop stuff out from the beginning of arrays.
*
*   Parameters : newData: the added input data.
*                newDataLength: The length of new data.
*                matchLength: The size of match string's lenght. Must be
*                             greater or equal to newDataLength.
*                windowHead: The current position of sliding window.
*                            Start to pop things out from here.
*                uncodedHead: The current position of uncoded Head.
*                
*   Effects    : "cuda_window" and "cuda_lookahead" are updated.
****************************************************************************/
void UpdateWindowAndLookAhead (const char *newData, int newDataLength
                               , int matchLength
                               ,int windowHead, int uncodedHead)
{
    int i = 0;
    while (i < newDataLength)
    {
        /* add old byte into sliding window and new into lookahead */
        slidingWindow[windowHead] = uncodedLookahead[uncodedHead];
        cudaMemcpy (cuda_window+windowHead, uncodedLookahead+uncodedHead
                    , sizeof(char), cudaMemcpyHostToDevice);
        uncodedLookahead[uncodedHead] = newData[i];

        cudaMemcpy (cuda_lookahead+uncodedHead, newData+i
                    , sizeof(char), cudaMemcpyHostToDevice);
        
        windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
        uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
        i++;
    }

    /* handle case where we hit EOF before filling lookahead */
    while (i < matchLength)
    {
        slidingWindow[windowHead] = uncodedLookahead[uncodedHead];
        cudaMemcpy (cuda_window+windowHead, uncodedLookahead+uncodedHead
                    , sizeof(char), cudaMemcpyHostToDevice);
        /* nothing to add to lookahead here */
        windowHead = Wrap((windowHead + 1), WINDOW_SIZE);
        uncodedHead = Wrap((uncodedHead + 1), MAX_CODED);
        i++;
    }
}
