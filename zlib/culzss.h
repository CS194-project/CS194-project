#ifndef ZLIB_CULZSS_H
#define ZLIB_CULZSS_H

typedef struct internal_state deflate_state;

typedef struct
{
  unsigned short dist;	/* offset to start of longest match */
  unsigned short len;	/* length of longest match */
} culzss_encoded_string_t;


const int CULZSS_WINDOW_BITS = 12;
const int CULZSS_WINDOW_SIZE = (1 << CULZSS_WINDOW_BITS);

const int CULZSS_HASH_SHIFT = 5;
const int CULZSS_HASH_BITS = 13;	/* HASH_SHIFT * MIN_MATCH >= HASH_BITS. */
const int CULZSS_HASH_SIZE = (1 << CULZSS_HASH_BITS);
const int CULZSS_HASH_MASK = (CULZSS_HASH_SIZE - 1);

const int CULZSS_LENGTH_BITS = 2;
const int CULZSS_MIN_MATCH = 3;
const int CULZSS_MAX_MATCH = (1 << CULZSS_LENGTH_BITS) + MIN_MATCH - 1;	/* 6  */

const int CULZSS_MAX_PROCESS_SIZE = (16 * 1024 * 1024);	/* Process 16M at one one. */
const int CULZSS_EXTRA_BUF = (2 * CULZSS_WINDOW_SIZE);	/* Extra BUF to avoid memory out of
                                             * bound. */

const int CULZSS_UNCODED = 1;
const int CULZSS_ENCODED = 0;

const int CULZSS_CUDA_BLOCK_SIZE = (1 * 1024 * 1024);	/* Size of bytes processed per kernel
                                                   launch. */
const int CULZSS_CUDA_NUM_BLOCKS = 1;	/* Max 4 blocks in GT740 if we run 1024 threads
                                   in each block. One kernel only runs in block
                                   in order to overlap kernel copy and execution. */

/* Number of streams. One stream must have a one-to-one relationship to a
   kernel instance. */
const int CULZSS_CUDA_NUM_STREAMS = (CULZSS_MAX_PROCESS_SIZE / CULZSS_CUDA_BLOCK_SIZE) + 1;

void culzss_init (deflate_state *s);
void culzss_destroy (deflate_state *s);
void culzss_longest_match (deflate_state *s);

#endif
