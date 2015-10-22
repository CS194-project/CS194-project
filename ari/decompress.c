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
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "modelA.h"
#include "decompressor.h"

int
main (int argc, char *argv[])
{
  if (argc < 3)
    {
      fprintf (stderr, "missing command line arguments\n");
      return 1;
    }
  FILE *input = fopen (argv[1], "rb");
  assert (input != NULL);
  FILE *output = fopen (argv[2], "wb");
  assert (output != NULL);
  struct modelA cmodel;
  model_init (&cmodel);

  model_dump ();
  printf ("decompressing...\n");
  decompress (input, output, &cmodel);
  printf ("%" PRIu64 "\n", cmodel.m_bytesProcessed);
  fclose (input);
  fclose (output);
  return 0;
}
