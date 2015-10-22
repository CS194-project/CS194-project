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
#ifndef BITIO_DOT_H
#define BITIO_DOT_H

#include <assert.h>
#include <stdio.h>

struct output_bits
{
  char m_NextByte;
  unsigned char m_Mask;
  FILE *m_Output;
};

void
output_bits_init (struct output_bits *ob, FILE * output)
{
  assert (ob != NULL);
  ob->m_Output = output;
  ob->m_NextByte = 0;
  ob->m_Mask = 0x80;
}

void
output_bits_finish (struct output_bits *ob)
{
  if (ob->m_Mask != 0x80)
    fputc ((int) ob->m_NextByte, ob->m_Output);
}

void
output_put_bit (struct output_bits *ob, int val)
{
  if (val)
    ob->m_NextByte |= ob->m_Mask;
  ob->m_Mask >>= 1;
  if (!ob->m_Mask)
    {
      fputc ((int) ob->m_NextByte, ob->m_Output);
      ob->m_Mask = 0x80;
      ob->m_NextByte = 0;
    }
}

struct input_bits
{
  unsigned char m_LastMask;
  int m_CurrentByte;
  int m_CodeValueBits;
  FILE *m_Input;
};

void
input_bits_init (struct input_bits *ib, FILE * input, int code_value_bits)
{
  assert (ib != NULL);
  ib->m_Input = input;
  ib->m_CurrentByte = 0;
  ib->m_LastMask = 1;
  ib->m_CodeValueBits = code_value_bits;
}

int
input_get_bit (struct input_bits *ib)
{
  if (ib->m_LastMask == 1)
    {
      ib->m_CurrentByte = fgetc (ib->m_Input);
      if (ib->m_CurrentByte < 0)
	{
	  if (ib->m_CodeValueBits <= 0)
	    {
	      fprintf (stderr, "EOF on input");
	      return 1;
	    }
	  else
	    ib->m_CodeValueBits -= 8;
	}
      ib->m_LastMask = 0x80;
    }
  else
    ib->m_LastMask >>= 1;
  return (ib->m_CurrentByte & ib->m_LastMask) != 0;
}

#endif //#ifndef BITIO_DOT_H
