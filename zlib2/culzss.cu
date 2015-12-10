#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/time.h>
#include <errno.h>
#include <assert.h>
#include <cuda.h>
#include <limits.h>
#include <algorithm>
#include "deflate.h"
#include "culzss.h"

void
checkCPUError (const char *msg)
{
    const int BUF_SIZE = 1024;
    char buf[BUF_SIZE];
    if (errno != 0 && errno != EEXIST)
    {
        char *errmsg = strerror_r (errno, buf, 1024);
        fprintf (stderr, "CPU error: %s: %s.\n", msg, errmsg);
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

void
culzss_init (deflate_state *s)
{
  /*
    for (int i = 0; i < CULZSS_CUDA_NUM_STREAMS; i++)
    {
        cudaStreamCreate (&(s->streams[i]));
        checkCudaError ("Cuda Create stream.");
        }*/

    /* Need initial WINDOW_SIZE bytes as initial window. EXTRA BUF to avoid out
     * of bound memory error. */
    // cudaHostAlloc (&s->host_in,
    //               sizeof (*s->host_in) * (CULZSS_MAX_PROCESS_SIZE + CULZSS_WINDOW_SIZE +
    //                                  CULZSS_EXTRA_BUF),
    //               cudaHostAllocDefault);
//checkCudaError ("Allocate host_in");
//
    s->host_in = (unsigned char*)malloc(
                   sizeof (*s->host_in) * (CULZSS_MAX_PROCESS_SIZE + CULZSS_WINDOW_SIZE +
                                               CULZSS_EXTRA_BUF));

    cudaMalloc (&s->device_in,
                sizeof (*s->device_in) * (CULZSS_MAX_PROCESS_SIZE + CULZSS_WINDOW_SIZE +
                                          CULZSS_EXTRA_BUF));
    checkCudaError ("Allocate device_in");

    s->host_encode = (culzss_encoded_string_t*)malloc(
                   sizeof (*s->host_encode) * (CULZSS_MAX_PROCESS_SIZE + CULZSS_WINDOW_SIZE +
                                               CULZSS_EXTRA_BUF));
    checkCudaError ("Allocate host_encode");
    cudaMalloc (&s->device_encode,
                sizeof (*s->device_encode) * (CULZSS_MAX_PROCESS_SIZE + CULZSS_WINDOW_SIZE +
                                              CULZSS_EXTRA_BUF));
    checkCudaError ("Allocate device_encode");
}


void
culzss_destroy (deflate_state *s)
{
    /* Clean up */
  /*
    for (int i = 0; i < CULZSS_CUDA_NUM_STREAMS; i++)
    {
        if (s->streams[i] != NULL)
            cudaStreamDestroy (s->streams[i]);
        s->streams[i] = NULL;
        }*/

    if (s->device_in != NULL)
        cudaFree (s->device_in);
    s->device_in = NULL;

    if (s->device_encode != NULL)
        cudaFree (s->device_encode);
    s->device_encode = NULL;

        if (s->host_in != NULL)
        cudaFreeHost (s->host_in);
     s->host_in = NULL;

    if (s->host_encode != NULL)
        free (s->host_encode);
    s->host_encode = NULL;
}

__global__ void
lzss_kernel (const unsigned char *__restrict__ in_g,
             culzss_encoded_string_t * __restrict__ encode, int grid_size,
             int is_firstblock)
{
    __shared__ unsigned char in[CULZSS_WINDOW_SIZE * 2];	/* Note that WINDOW_SIZE must be a
                                                               multiple of blockDimension(1024).
                                                               First half are window and second half are
                                                               lookahead. */
    __shared__ unsigned short hashtable[CULZSS_HASH_SIZE];
    const int CULZSS_STEP_SIZE = 64 * 1024;
    /* We do 64KB every step. (Actual 64KB - 4KB (window size) because of
       overlapping).
       This is because we use unsigned short as the
       type of hashtable to maximize the number of indexes in hashtable.
       It can store the indexes to the maximum to 65535. */

    while (grid_size > 0)
    {
        int block_size = grid_size - (CULZSS_STEP_SIZE * blockIdx.x -
                                      CULZSS_WINDOW_SIZE * blockIdx.x);
        if (block_size > CULZSS_STEP_SIZE)
            block_size = CULZSS_STEP_SIZE;
        encode += (CULZSS_STEP_SIZE - CULZSS_WINDOW_SIZE) * blockIdx.x;
        in_g += (CULZSS_STEP_SIZE - CULZSS_WINDOW_SIZE) * blockIdx.x;

        /* initialize shared memory. */
        /* Note that in_g needs to have some extra space at the end to avoid invalid
           memory access */
        for (int i = threadIdx.x * 4; i < CULZSS_WINDOW_SIZE * 2; i += blockDim.x * 4)
        {
            *((int *) (in + i)) = *((int *) (in_g + i));
        }
        __syncthreads ();

        /* Compute hash of initial sliding window. */
        for (int ii = threadIdx.x * 4; ii < CULZSS_WINDOW_SIZE; ii += blockDim.x * 4)
        {
            for (int i = ii; i < ii + 4; i++)
            {
                int hash = 0;
                hash = (((hash) << CULZSS_HASH_SHIFT) ^ (in[i]));
                hash = (((hash) << CULZSS_HASH_SHIFT) ^ (in[i + 1]));
                hash = (((hash) << CULZSS_HASH_SHIFT) ^ (in[i + 2]));
                hash = hash & CULZSS_HASH_MASK;
                hashtable[hash] = i;
            }
        }
        __syncthreads ();

        int hash0 = 0, index0 = threadIdx.x;
        int hash1 = 0, index1 = threadIdx.x;
        int hash2 = 0, index2 = threadIdx.x;
        int hash3 = 0, index3 = threadIdx.x;
        for (int uncodedHead = CULZSS_WINDOW_SIZE; uncodedHead < block_size;
             uncodedHead += CULZSS_WINDOW_SIZE)
        {
            int end = CULZSS_WINDOW_SIZE + min (block_size - uncodedHead, CULZSS_WINDOW_SIZE);

            for (int i = threadIdx.x + CULZSS_WINDOW_SIZE; i < 2 * CULZSS_WINDOW_SIZE;
                 i += blockDim.x)
            {
                unsigned int hash = 0;
                unsigned char char0 = in[i];
                unsigned char char1 = 0;
                unsigned char char2 = 0;
                unsigned char char3 = 0;
                unsigned char char4 = 0;
                unsigned char char5 = 0;
                culzss_encoded_string_t match_data;
                int match_len;
                unsigned int prev;
                match_data.len = 0;
                match_data.dist = 0;

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
                    hash = (((hash) << CULZSS_HASH_SHIFT) ^ (char0));
                    hash = (((hash) << CULZSS_HASH_SHIFT) ^ (char1));
                    hash = (((hash) << CULZSS_HASH_SHIFT) ^ (char2));
                    hash = hash & CULZSS_HASH_MASK;
                }

                /* Check most recent hash. */
                __syncthreads ();
                prev = hashtable[hash];
                prev = prev + CULZSS_WINDOW_SIZE - uncodedHead;
                match_len = 0;

                if (prev > 0
                    && prev < i - CULZSS_MAX_MATCH + 1
                    && i - prev <= CULZSS_WINDOW_SIZE && i < end - CULZSS_MAX_MATCH + 1)
                {
                    int temp = 1;	/* Used to check match. */
                    temp *= (in[prev] == char0);
                    match_len += temp;
                    temp *= (in[prev + 1] == char1);
                    match_len += temp;
                    temp *= (in[prev + 2] == char2);
                    match_len += temp;
                    temp *= (in[prev + 3] == char3);
                    match_len += temp;
                    temp *= (in[prev + 4] == char4);
                    match_len += temp;
                    temp *= (in[prev + 5] == char5);
                    match_len += temp;
                }
                if (match_len > match_data.len)
                {
                    match_data.dist = i - prev;
                    match_data.len = match_len;
                }

                /* Check second recent hash. */
                __syncthreads ();
                hashtable[hash1] = index1;
                __syncthreads ();

                prev = hashtable[hash];
                prev = prev + CULZSS_WINDOW_SIZE - uncodedHead;
                match_len = 0;
                if (prev > 0
                    && prev < i - CULZSS_MAX_MATCH + 1
                    && i - prev <= CULZSS_WINDOW_SIZE && i < end - CULZSS_MAX_MATCH + 1)
                {
                    int temp = 1;	/* Used to check match. */
                    temp *= (in[prev] == char0);
                    match_len += temp;
                    temp *= (in[prev + 1] == char1);
                    match_len += temp;
                    temp *= (in[prev + 2] == char2);
                    match_len += temp;
                    temp *= (in[prev + 3] == char3);
                    match_len += temp;
                    temp *= (in[prev + 4] == char4);
                    match_len += temp;
                    temp *= (in[prev + 5] == char5);
                    match_len += temp;
                }
                if (match_len > match_data.len)
                {
                    match_data.dist = i - prev;
                    match_data.len = match_len;
                }


                /* Check third recent hash. */
                __syncthreads ();
                hashtable[hash2] = index2;
                __syncthreads ();

                prev = hashtable[hash];
                prev = prev + CULZSS_WINDOW_SIZE - uncodedHead;
                match_len = 0;
                if (prev > 0
                    && prev < i - CULZSS_MAX_MATCH + 1
                    && i - prev <= CULZSS_WINDOW_SIZE && i < end - CULZSS_MAX_MATCH + 1)
                {
                    int temp = 1;	/* Used to check match. */
                    temp *= (in[prev] == char0);
                    match_len += temp;
                    temp *= (in[prev + 1] == char1);
                    match_len += temp;
                    temp *= (in[prev + 2] == char2);
                    match_len += temp;
                    temp *= (in[prev + 3] == char3);
                    match_len += temp;
                    temp *= (in[prev + 4] == char4);
                    match_len += temp;
                    temp *= (in[prev + 5] == char5);
                    match_len += temp;
                }
                if (match_len > match_data.len)
                {
                    match_data.dist = i - prev;
                    match_data.len = match_len;
                }

                /* Check forth recent hash. */
                __syncthreads ();
                hashtable[hash3] = index3;
                __syncthreads ();

                prev = hashtable[hash];
                prev = prev + CULZSS_WINDOW_SIZE - uncodedHead;
                match_len = 0;
                if (prev > 0
                    && prev < i - CULZSS_MAX_MATCH + 1 && i - prev <= CULZSS_WINDOW_SIZE
                    && i < end - CULZSS_MAX_MATCH + 1)
                {
                    int temp = 1;	/* Used to check match. */
                    temp *= (in[prev] == char0);
                    match_len += temp;
                    temp *= (in[prev + 1] == char1);
                    match_len += temp;
                    temp *= (in[prev + 2] == char2);
                    match_len += temp;
                    temp *= (in[prev + 3] == char3);
                    match_len += temp;
                    temp *= (in[prev + 4] == char4);
                    match_len += temp;
                    temp *= (in[prev + 5] == char5);
                    match_len += temp;
                }
                if (match_len > match_data.len)
                {
                    match_data.dist = i - prev;
                    match_data.len = match_len;
                }

                /* Update recent hash */
                hash3 = hash2;
                index3 = index2;
                hash2 = hash1;
                index2 = index1;
                hash1 = hash0;
                index1 = index0;
                hash0 = hash0;
                index0 = i + uncodedHead - CULZSS_WINDOW_SIZE;

                /* Check current hash. */
                __syncthreads ();
                hashtable[hash] = index0;
                __syncthreads ();

                prev = hashtable[hash];
                prev = prev + CULZSS_WINDOW_SIZE - uncodedHead;
                match_len = 0;
                if (prev > 0
                    && prev < i - CULZSS_MAX_MATCH + 1 && i - prev <= CULZSS_WINDOW_SIZE
                    && i < end - CULZSS_MAX_MATCH + 1)
                {
                    int temp = 1;	/* Used to check match. */
                    temp *= (in[prev] == char0);
                    match_len += temp;
                    temp *= (in[prev + 1] == char1);
                    match_len += temp;
                    temp *= (in[prev + 2] == char2);
                    match_len += temp;
                    temp *= (in[prev + 3] == char3);
                    match_len += temp;
                    temp *= (in[prev + 4] == char4);
                    match_len += temp;
                    temp *= (in[prev + 5] == char5);
                    match_len += temp;
                }

                if (match_len > match_data.len)
                {
                    match_data.dist = i - prev;
                    match_data.len = match_len;
                }

                /* Don't compress first block because initial window can be arbitrary. */
                if ((blockIdx.x == 0 && uncodedHead == CULZSS_WINDOW_SIZE
                     && is_firstblock) || match_data.len < CULZSS_MIN_MATCH)
                {
                    match_data.dist = 0;
                    match_data.len = (unsigned short)char0; /* Store string literal in
                                                               match_data.len. (We know its a literal by match_data.dist
                                                               == 0) */
                }
                /* Done at this position. Store to global memory. */
                if (i < end)
                {
                    encode[i + uncodedHead - CULZSS_WINDOW_SIZE] = match_data;
                }
            }

            __syncthreads ();

            /* Move sliding window. */
            for (int j = threadIdx.x * 4; j < CULZSS_WINDOW_SIZE; j += blockDim.x * 4)
            {
                *((int *) (in + j)) = *((int *) (in + j + CULZSS_WINDOW_SIZE));
            }

            for (int j = threadIdx.x * 4; j < CULZSS_WINDOW_SIZE; j += blockDim.x * 4)
            {
                *((int *) (in + CULZSS_WINDOW_SIZE + j)) =
                    *((int *) (in_g + uncodedHead + CULZSS_WINDOW_SIZE + j));
            }

            __syncthreads ();
        }

        grid_size -= CULZSS_STEP_SIZE - CULZSS_WINDOW_SIZE;
        encode += (CULZSS_STEP_SIZE - CULZSS_WINDOW_SIZE) * gridDim.x;
        in_g += (CULZSS_STEP_SIZE - CULZSS_WINDOW_SIZE) * gridDim.x;
    }
}


/* deflate_state must have been initialized. */
int
culzss_longest_match (deflate_state *s, int size, int is_firstblock, int flush)
{
  cudaMemcpyAsync (s->device_in + CULZSS_WINDOW_SIZE,
                   s->host_in + CULZSS_WINDOW_SIZE,
                   size,
                   cudaMemcpyHostToDevice);
  checkCudaError ("copy from host_in to device_in");

  lzss_kernel <<< CULZSS_CUDA_NUM_BLOCKS, 1024, 0,
    NULL >>> (s->device_in, s->device_encode, size+CULZSS_WINDOW_SIZE,
                                 is_firstblock);
  checkCudaError ("launch lzss_kernel.");

  cudaMemcpyAsync (s->host_encode + CULZSS_WINDOW_SIZE,
                   s->device_encode + CULZSS_WINDOW_SIZE,
                   size,
                   cudaMemcpyDeviceToHost, NULL);
  checkCudaError ("Copy from device_on to host_in.");

  cudaDeviceSynchronize ();
    int bflush;			/* set if current block must be flushed */

    int cur = 0;
    for (;;)
    {
        if (s->lookahead < MIN_LOOKAHEAD)
        {
            fill_window (s);
            if (s->lookahead < MIN_LOOKAHEAD && flush == Z_NO_FLUSH)
            {
                return need_more;
            }
            if (s->lookahead == 0)
                break;		/* flush the current block */


            if (s->lookahead < MIN_LOOKAHEAD)
            {
                Tracevv ((stderr, "%c", s->window[s->strstart]));
                _tr_tally_lit (s, s->window[s->strstart], bflush);
                s->lookahead--;
                s->strstart++;
                cur++;

                if (bflush)
                    FLUSH_BLOCK (s, 0);
            }
        }

        else
        {
            while (s->lookahead >= MIN_LOOKAHEAD)
            {
                if (cur < CULZSS_WINDOW_SIZE || s->host_encode[cur].dist == 0
                    || s->lookahead - s->host_encode[cur].len <= MIN_LOOKAHEAD)
                {
                    _tr_tally_lit (s, s->window[s->strstart], bflush);
                    s->strstart++;
                    cur++;
                    if (s->lookahead == 0)
                        break;
                    s->lookahead--;
                }
                else
                {
                    int match_length = s->host_encode[cur].len;
                    s->match_length = match_length>s->lookahead? s->lookahead: match_length;
                    _tr_tally_dist (s, s->host_encode[cur].dist,
                                    s->match_length - MIN_MATCH, bflush);
                    s->strstart += s->match_length;
                    cur += s->match_length;
                    s->lookahead -= s->match_length;
                    s->match_length = 0;
                    s->ins_h = s->window[s->strstart];
                }
                if (bflush)
                    FLUSH_BLOCK (s, 0);
            }
        }
    }

    s->insert = s->strstart < MIN_MATCH - 1 ? s->strstart : MIN_MATCH - 1;
    if (flush == Z_FINISH)
    {
        FLUSH_BLOCK (s, 1);
        return finish_done;
    }
    if (s->last_lit)
        FLUSH_BLOCK (s, 0);
    return block_done;
}
