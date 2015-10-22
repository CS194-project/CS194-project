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
#ifndef MODEL_A_DOT_H
#define MODEL_A_DOT_H

#include <stdio.h>
#include <assert.h>
#include <inttypes.h>

//
// By default we set this whole thing up so that 
// all math is done IN Model_code_type integers, which
// are usually 32 bit ints or longs. In these
// you might set things up so that your max
// frequency or your max code value fit into
// a smaller size integer, say 16 bits. You can
// possibly get some efficiency then by storing
// counts in a smaller type of integer, but at this
// time we are not going to try to implement that.
// If we were going to do so automatically, this 
// template taken from this stackoverflow convo:
// http://stackoverflow.com/questions/12082571/how-to-figure-out-the-smallest-integral-type-that-can-represent-a-number-in-com
//
typedef uint64_t Model_code_type;
struct Model_prob
{
  Model_code_type low;
  Model_code_type high;
  Model_code_type count;
};

#define MODEL_NAME "cmodel"
#define MODEL_PRECISION            ((int)(sizeof(Model_code_type)*8))
#define MODEL_CODE_VALUE_BITS      ((MODEL_PRECISION + 3) / 2)
#define MODEL_FREQUENCY_BITS     (MODEL_PRECISION - MODEL_CODE_VALUE_BITS)
#define MODEL_MAX_CODE      ((Model_code_type)(((Model_code_type)(1) << MODEL_CODE_VALUE_BITS) - 1))
#define MODEL_MAX_FREQ      ((Model_code_type)(((Model_code_type)(1) << MODEL_FREQUENCY_BITS) - 1))
#define MODEL_ONE_FOURTH    ((Model_code_type)((Model_code_type)(1) << (MODEL_CODE_VALUE_BITS - 2)))
#define MODEL_ONE_HALF      (2 * MODEL_ONE_FOURTH)
#define MODEL_THREE_FOURTHS (3 * MODEL_ONE_FOURTH)

struct modelA
{
  int m_frozen;
  uint64_t m_bytesProcessed;
  //
  // variables used by the model
  //
  Model_code_type cumulative_frequency[258];	//Character a is defined by the range cumulative_frequency[a],
  //cumulative_frequency[a+1], with cumulative_frequency[257]
  //containing the total count for the model. Note that entry
  //256 is the EOF.
};
void
model_init (struct modelA *m)
{
  assert (MODEL_PRECISION >= MODEL_CODE_VALUE_BITS);
  assert (MODEL_FREQUENCY_BITS <= (MODEL_CODE_VALUE_BITS + 2));
  assert ((MODEL_CODE_VALUE_BITS + MODEL_FREQUENCY_BITS) <= MODEL_PRECISION);
  assert (MODEL_MAX_FREQ > 257);
  for (int i = 0; i < 258; i++)
    m->cumulative_frequency[i] = i;
  m->m_bytesProcessed = 0;
  m->m_frozen = 0;
}

inline void
model_pacify (struct modelA *m)
{
  if ((++m->m_bytesProcessed % 1000) == 0)
    printf ("%" PRIu64 "\r", m->m_bytesProcessed);
}

inline void
model_frozen (const struct modelA *m)
{
  printf ("Frozen at: %" PRIu64 "\n", m->m_bytesProcessed);
}

inline void
model_update (struct modelA *m, int c)
{
  for (int i = c + 1; i < 258; i++)
    m->cumulative_frequency[i]++;
  if (m->cumulative_frequency[257] >= MODEL_MAX_FREQ)
    {
      m->m_frozen = 1;
      model_frozen (m);
    }
}

struct Model_prob
model_get_probability (struct modelA *m, int c)
{
  struct Model_prob p =
    { m->cumulative_frequency[c], m->cumulative_frequency[c + 1],
m->cumulative_frequency[257] };
  if (!m->m_frozen)
    model_update (m, c);
  model_pacify (m);
  return p;
}

struct Model_prob
model_getchar (struct modelA *m, Model_code_type scaled_value, int *c)
{
  model_pacify (m);
  for (int i = 0; i < 257; i++)
    if (scaled_value < m->cumulative_frequency[i + 1])
      {
	*c = i;
	struct Model_prob p =
	  { m->cumulative_frequency[i], m->cumulative_frequency[i + 1],
m->cumulative_frequency[257] };
	if (!m->m_frozen)
	  model_update (m, *c);
	return p;
      }
  assert (0);
}

Model_code_type
model_getcount (const struct modelA * m)
{
  return m->cumulative_frequency[257];
}

void
model_dump ()
{
  fprintf (stderr,
	   "Model %s\n"
	   "PRECISION %d bits \n"
	   "CODE_VALUE_BITS %d bits giving MAX_CODE of %" PRIu64 "\n"
	   "FREQUENCY_BITS %d bits giving MAX_FREQUENCY of %" PRIu64 "\n"
	   "MAX_CODE: %" PRIu64 " %#018" PRIx64 "\n"
	   "MAX_FREQ: %" PRIu64 " %#018" PRIx64 "\n"
	   "ONE_FOURTH: %" PRIu64 " %#018" PRIx64 "\n"
	   "ONE_HALF: %" PRIu64 " %#018" PRIx64 "\n"
	   "THREE_FOURTHS: %" PRIu64 " %#018" PRIx64 "\n", MODEL_NAME,
	   MODEL_PRECISION, MODEL_CODE_VALUE_BITS, MODEL_MAX_CODE,
	   MODEL_FREQUENCY_BITS, MODEL_MAX_FREQ, MODEL_MAX_CODE,
	   MODEL_MAX_CODE, MODEL_MAX_FREQ, MODEL_MAX_FREQ, MODEL_ONE_FOURTH,
	   MODEL_ONE_FOURTH, MODEL_ONE_HALF, MODEL_ONE_HALF,
	   MODEL_THREE_FOURTHS, MODEL_THREE_FOURTHS);
}

#endif //#ifndef MODEL_A_DOT_H
