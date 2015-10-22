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
#ifndef DECOMPESSOR_DOT_H
#define DECOMPESSOR_DOT_H

#ifdef LOG
#include <stdio.h>
#endif

#include <inttypes.h>
#include "bitio.h"

//
// The arithmetic decompressor is a general purpose decompressor that
// is parameterized on the types of the input, output, and 
// model objects, in an attempt to make it as flexible as
// possible. It is easiest to use by calling the compress()
// convenience function found at the bottom of this header file
//
// The INPUT class is expected to provide a get_bit() function,
// while the output function is expected to provider a put_byte()
// function. Both of these functions should throw exceptions on
// errors. We expect the EOF to be embedded in the compressed
// stream, so it needs to be extracted by the decoder. If the
// compression goes awry, the get_bit() function will be 
// repeatedly called on EOF(), in which case it would be good
// for it to return an error.
//

int
decompress (FILE * input, FILE * m_output, struct modelA *m_model)
{
  struct input_bits m_input;
  input_bits_init (&m_input, input, MODEL_CODE_VALUE_BITS);
#ifdef LOG
  FILE *log = fopen ("decompressor.log", "w");
  assert (log != NULL);
#endif
  Model_code_type high = MODEL_MAX_CODE;
  Model_code_type low = 0;
  Model_code_type value = 0;
  for (int i = 0; i < MODEL_CODE_VALUE_BITS; i++)
    {
      value <<= 1;
      value += input_get_bit (&m_input) ? 1 : 0;
    }
  for (;;)
    {
      int ret;
      Model_code_type range = high - low + 1;
      Model_code_type scaled_value =
	((value - low + 1) * model_getcount (m_model) - 1) / range;
      int c;
      struct Model_prob p = model_getchar (m_model, scaled_value, &c);
      if (c == 256)
	break;
      ret = fputc (c, m_output);
      assert (ret >= 0);
#ifdef LOG
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
      high = low + (range * p.high) / p.count - 1;
      low = low + (range * p.low) / p.count;
#ifdef LOG
      ret = fprintf (log, "%#018" PRIx64 "  %#018" PRIx64 "\n", low, high);
      assert (ret > 0);
#endif
      for (;;)
	{
	  if (high < MODEL_ONE_HALF)
	    {
	      //do nothing, bit is a zero
	    }
	  else if (low >= MODEL_ONE_HALF)
	    {
	      value -= MODEL_ONE_HALF;	//subtract one half from all three code values
	      low -= MODEL_ONE_HALF;
	      high -= MODEL_ONE_HALF;
	    }
	  else if (low >= MODEL_ONE_FOURTH && high < MODEL_THREE_FOURTHS)
	    {
	      value -= MODEL_ONE_FOURTH;
	      low -= MODEL_ONE_FOURTH;
	      high -= MODEL_ONE_FOURTH;
	    }
	  else
	    break;
	  low <<= 1;
	  high <<= 1;
	  high++;
	  value <<= 1;
	  value += input_get_bit (&m_input) ? 1 : 0;
	}
    }
#ifdef LOG
  int ret;
  ret = fprintf (log, "%#04x", 256);
  assert (ret > 0);
  ret = fprintf (log, " %#018" PRIx64 "  %#018" PRIx64 " => ", low, high);
  assert (ret > 0);
  fclose (log);
#endif
  return 0;
}

#endif //#ifndef DECOMPESSOR_DOT_H
