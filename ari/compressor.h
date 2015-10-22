/*
The MIT License (MIT)

Copyright (c) 2014 Mark Thomas Nelson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

 This code was written to illustrate the article:
 Data Compression With Arithmetic Coding
 by Mark Nelson
 published at: http://marknelson.us/2014/10/19/data-compression-with-arithmetic-coding

*/
#ifndef COMPRESSOR_DOT_H
#define COMPRESSOR_DOT_H

#include "bitio.h"
#include "assert.h"
#include <inttypes.h>
#ifdef LOG
#include <stdio.h>
#endif

inline void
put_bit_plus_pending (struct output_bits *m_output, int bit,
		      int *pending_bits)
{
  output_put_bit (m_output, bit);
  for (int i = 0; i < *pending_bits; i++)
    output_put_bit (m_output, !bit);
  *pending_bits = 0;
}

int
compress (FILE * m_input, FILE * output, struct modelA *m_model)
{
  struct output_bits m_output;
  output_bits_init (&m_output, output);
#ifdef LOG
  FILE *log = fopen ("compressor.log", "w");
  assert (log != NULL);
#endif
  int pending_bits = 0;
  Model_code_type low = 0;
  Model_code_type high = MODEL_MAX_CODE;
  for (;;)
    {
      int c = fgetc (m_input);
      if (c == -1)
	c = 256;
#ifdef LOG
      int ret;
      ret = fprintf (log, "%#04x", c);
      assert (ret > 0);
      if (c > 0x20 && c <= 0x7f)
	{
	  ret = fprintf (log, "(%c)", char (c));
	  assert (ret > 0);
	}
      else
	{
	  ret = fprintf (log, "   ");
	  assert (ret > 0);
	}
      ret = fprintf (log, " %#018" PRIx64 "  %#018" PRIx64 " => ", low, high);
      assert (ret > 0);
#endif
      struct Model_prob p = model_get_probability (m_model, c);
      Model_code_type range = high - low + 1;
      high = low + (range * p.high / p.count) - 1;
      low = low + (range * p.low / p.count);
#ifdef LOG
      ret = fprintf (log, "%#018" PRIx64 "  %#018" PRIx64 "\n", low, high);
      assert (ret > 0);
#endif
      //
      // On each pass there are six possible configurations of high/low,
      // each of which has its own set of actions. When high or low
      // is converging, we output their MSB and upshift high and low.
      // When they are in a near-convergent state, we upshift over the
      // next-to-MSB, increment the pending count, leave the MSB intact,
      // and don't output anything. If we are not converging, we do
      // no shifting and no output.
      // high: 0xxx, low anything : converging (output 0)
      // low: 1xxx, high anything : converging (output 1)
      // high: 10xxx, low: 01xxx : near converging
      // high: %11xxx, low: 01xxx : not converging
      // high: 11xxx, low: 00xxx : not converging
      // high: 10xxx, low: 00xxx : not converging
      //
      for (;;)
	{
	  if (high < MODEL_ONE_HALF)
	    put_bit_plus_pending (&m_output, 0, &pending_bits);
	  else if (low >= MODEL_ONE_HALF)
	    put_bit_plus_pending (&m_output, 1, &pending_bits);
	  else if (low >= MODEL_ONE_FOURTH && high < MODEL_THREE_FOURTHS)
	    {
	      pending_bits++;
	      low -= MODEL_ONE_FOURTH;
	      high -= MODEL_ONE_FOURTH;
	    }
	  else
	    break;
	  high <<= 1;
	  high++;
	  low <<= 1;
	  high &= MODEL_MAX_CODE;
	  low &= MODEL_MAX_CODE;
	}
      if (c == 256)		//256 is the special EOF code
	break;
    }
  pending_bits++;
  if (low < MODEL_ONE_FOURTH)
    put_bit_plus_pending (&m_output, 0, &pending_bits);
  else
    put_bit_plus_pending (&m_output, 1, &pending_bits);
  output_bits_finish (&m_output);
#ifdef LOG
  fclose (log);
#endif
  return 0;
}


#endif //#ifndef COMPRESSOR_DOT_H
