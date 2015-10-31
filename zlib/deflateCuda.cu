#include "deflate.h"
#include <string.h>
#include <stdio.h>

/* ===========================================================================
 * Set match_start to the longest match starting at the given string and
 * return its length. Matches shorter or equal to prev_length are discarded,
 * in which case the result is equal to prev_length and match_start is
 * garbage.
 * IN assertions: cur_match is the head of the hash chain for the current
 *   string (strstart) and its distance is <= MAX_DIST, and prev_length >= 1
 * OUT assertion: the match length is not greater than s->lookahead.
 */
/* #ifndef ASMV */ /* Disabled */
 /* For 80x86 and 680x0, an optimized version will be provided in match.asm or
 * match.S. The code will be functionally equivalent.
 */
__global__ void
longest_match_ (deflate_state * s, IPos cur_match, int *cuda_return, Bytef *window
                , Posf *prev)
{
  unsigned chain_length = s->max_chain_length;	/* max hash chain length */
  register Bytef *scan = window + s->strstart;	/* current string */
  register Bytef *match;	/* matched string */
  register int len;		/* length of current match */
  int best_len = s->prev_length;	/* best match length so far */
  int nice_match = s->nice_match;	/* stop if match long enough */
  IPos limit = s->strstart > (IPos) MAX_DIST (s) ?
    s->strstart - (IPos) MAX_DIST (s) : Z_NULL;
  /* Stop when cur_match becomes <= limit. To simplify the code,
   * we prevent matches with the string of window index 0.
   */
  //Posf *prev = s->prev;
  uInt wmask = s->w_mask;

#ifdef UNALIGNED_OK
  /* Compare two bytes at a time. Note: this is not always beneficial.
   * Try with and without -DUNALIGNED_OK to check.
   */
  register Bytef *strend = s->window + s->strstart + MAX_MATCH - 1;
  register ush scan_start = *(ushf *) scan;
  register ush scan_end = *(ushf *) (scan + best_len - 1);
#else
  register Bytef *strend = s->window + s->strstart + MAX_MATCH;
  register Byte scan_end1 = scan[best_len - 1];
  register Byte scan_end = scan[best_len];
#endif

  /* The code is optimized for HASH_BITS >= 8 and MAX_MATCH-2 multiple of 16.
   * It is easy to get rid of this optimization if necessary.
   */
  /*Assert (s->hash_bits >= 8 && MAX_MATCH == 258, "Code too clever");*/

  /* Do not waste too much time if we already have a good match: */
  if (s->prev_length >= s->good_match)
    {
      chain_length >>= 2;
    }
  /* Do not look for matches beyond the end of the input. This is necessary
   * to make deflate deterministic.
   */
  if ((uInt) nice_match > s->lookahead)
    nice_match = s->lookahead;

  /* Assert ((ulg) s->strstart <= s->window_size - MIN_LOOKAHEAD,
     "need lookahead"); */

  do
    {
        /*  Assert (cur_match < s->strstart, "no future"); */
      match = s->window + cur_match;

      /* Skip to next match if the match length cannot increase
       * or if the match length is less than 2.  Note that the checks below
       * for insufficient lookahead only occur occasionally for performance
       * reasons.  Therefore uninitialized memory will be accessed, and
       * conditional jumps will be made that depend on those values.
       * However the length of the match is limited to the lookahead, so
       * the output of deflate is not affected by the uninitialized values.
       */
#if (defined(UNALIGNED_OK) && MAX_MATCH == 258)
      /* This code assumes sizeof(unsigned short) == 2. Do not use
       * UNALIGNED_OK if your compiler uses a different size.
       */
      if (*(ushf *) (match + best_len - 1) != scan_end ||
	  *(ushf *) match != scan_start)
	continue;

      /* It is not necessary to compare scan[2] and match[2] since they are
       * always equal when the other bytes match, given that the hash keys
       * are equal and that HASH_BITS >= 8. Compare 2 bytes at a time at
       * strstart+3, +5, ... up to strstart+257. We check for insufficient
       * lookahead only every 4th comparison; the 128th check will be made
       * at strstart+257. If MAX_MATCH-2 is not a multiple of 8, it is
       * necessary to put more guard bytes at the end of the window, or
       * to check more often for insufficient lookahead.
       */
      /*  Assert (scan[2] == match[2], "scan[2]?"); */
      scan++, match++;
      do
	{
	}
      while (*(ushf *) (scan += 2) == *(ushf *) (match += 2) &&
	     *(ushf *) (scan += 2) == *(ushf *) (match += 2) &&
	     *(ushf *) (scan += 2) == *(ushf *) (match += 2) &&
	     *(ushf *) (scan += 2) == *(ushf *) (match += 2) &&
	     scan < strend);
      /* The funny "do {}" generates better code on most compilers */

      /* Here, scan <= window+strstart+257 */
      /*   Assert (scan <= s->window + (unsigned) (s->window_size - 1),
           "wild scan"); */
      if (*scan == *match)
	scan++;

      len = (MAX_MATCH - 1) - (int) (strend - scan);
      scan = strend - (MAX_MATCH - 1);

#else /* UNALIGNED_OK */

      if (match[best_len] != scan_end ||
	  match[best_len - 1] != scan_end1 ||
	  *match != *scan || *++match != scan[1])
	continue;

      /* The check at best_len-1 can be removed because it will be made
       * again later. (This heuristic is not always a win.)
       * It is not necessary to compare scan[2] and match[2] since they
       * are always equal when the other bytes match, given that
       * the hash keys are equal and that HASH_BITS >= 8.
       */
      scan += 2, match++;
      /* Assert (*scan == *match, "match[2]?"); */

      /* We check for insufficient lookahead only every 8th comparison;
       * the 256th check will be made at strstart+258.
       */
      do
	{
	}
      while (*++scan == *++match && *++scan == *++match &&
	     *++scan == *++match && *++scan == *++match &&
	     *++scan == *++match && *++scan == *++match &&
	     *++scan == *++match && *++scan == *++match && scan < strend);

      /* Assert (scan <= s->window + (unsigned) (s->window_size - 1),
         "wild scan");*/

      len = MAX_MATCH - (int) (strend - scan);
      scan = strend - MAX_MATCH;

#endif /* UNALIGNED_OK */

      if (len > best_len)
	{
	  s->match_start = cur_match;
	  best_len = len;
	  if (len >= nice_match)
	    break;
#ifdef UNALIGNED_OK
	  scan_end = *(ushf *) (scan + best_len - 1);
#else
	  scan_end1 = scan[best_len - 1];
	  scan_end = scan[best_len];
#endif
	}
    }
  while ((cur_match = prev[cur_match & wmask]) > limit
	 && --chain_length != 0);

  if ((uInt) best_len <= s->lookahead)
    *cuda_return = (uInt) best_len;
  else
      *cuda_return = s->lookahead;
}

int longest_match_cuda(deflate_state * s, IPos cur_match)
{
    deflate_state *state;
    Bytef *window;
    Posf *prev;
    int *cuda_return;
    cudaMalloc(&state, sizeof(deflate_state));
    cudaMalloc(&prev, s->w_size*sizeof(Posf));
    cudaMalloc(&window, s->w_size*2*sizeof(Bytef));
    cudaMalloc(&cuda_return, sizeof(int));
    
    cudaMemcpy(prev, s->prev, s->w_size*sizeof(Posf), cudaMemcpyHostToDevice);
    cudaMemcpy(state, s, sizeof(deflate_state), cudaMemcpyHostToDevice);
    cudaMemcpy(window, s->window,s->w_size*2*sizeof(Bytef)
               , cudaMemcpyHostToDevice);
    
    longest_match_<<<1,1>>>(state, cur_match, cuda_return, window, prev);
    
    int host_return[1];
    
    cudaMemcpy (host_return, cuda_return, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy (s, state, sizeof(deflate_state), cudaMemcpyDeviceToHost);
    
    cudaFree(state);
    cudaFree(prev);
    cudaFree(window);
    return host_return[0];
}
