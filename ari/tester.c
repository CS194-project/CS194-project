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
#include <assert.h>

#include "modelA.h"
#include "compressor.h"
#include "decompressor.h"

int validate (const char *input_file,
	      const char *compressed_file,
	      const char *output_file, double *bpb);

int
main (int argc, char *argv[])
{
  if (argc < 2)
    {
      fprintf (stderr, "missing command line arguments\n");
      return 1;
    }
  printf ("compressing %s ...\n", argv[1]);
  FILE *input1 = fopen (argv[1], "rb");
  assert (input1 != NULL);
  FILE *output1 = fopen ("temp.ari", "wb");
  assert (output1 != NULL);
  struct modelA cmodel;

  model_init (&cmodel);
  compress (input1, output1, &cmodel);
  fclose (input1);
  fclose (output1);

  FILE *input2 = fopen ("temp.ari", "rb");
  assert (input2 != NULL);
  FILE *output2 = fopen ("temp.out", "wb");
  assert (output2 != NULL);

  model_init (&cmodel);
  decompress (input2, output2, &cmodel);
  fclose (input2);
  fclose (output2);

  double bpb;
  validate (argv[1], "temp.ari", "temp.out", &bpb);
  printf ("%lf\n", bpb);
  return 0;
}

int
validate (const char *input_file,
	  const char *compressed_file, const char *output_file, double *bpb)
{
  int verbose = 1;		//might turn this on in some contexts

  FILE *in = fopen (input_file, "rb");
  if (!in)
    {
      fprintf (stderr, "validate error opening inptut file: %s\n",
	       input_file);
      return 1;
    }
  FILE *compressed = fopen (compressed_file, "rb");
  if (!compressed)
    {
      fprintf (stderr, "validate error opening compressed file: %s\n",
	       compressed_file);
      return 1;
    }

  FILE *out = fopen (output_file, "rb");
  if (!out)
    {
      fprintf (stderr, "validate error opening output file: %s\n",
	       output_file);
      return 1;
    }
  fseek (in, 0, SEEK_END);
  fseek (out, 0, SEEK_END);
  fseek (compressed, 0, SEEK_END);
  long in_length = ftell (in);
  long out_length = ftell (out);
  long compressed_length = ftell (compressed);
  fseek (in, 0, SEEK_SET);
  fseek (out, 0, SEEK_SET);
  fseek (compressed, 0, SEEK_SET);
  if (verbose)
    printf ("input length: %ld\noutput length: %ld\ncompressed length: %ld\n",
	    in_length, out_length, compressed_length);
  if (in_length != out_length)
    {
      fprintf (stderr,
	       "Error, input file and output file have different lengths\n");
      return 1;
    }
  if ((long long) (in_length) == 0)
    *bpb = 8.0;
  else
    *bpb = compressed_length * 8.0 / in_length;
  if (verbose)
    printf ("Compressed to %lf bits per byte\n", *bpb);
  int c1;
  int c2;
  while (c1 = fgetc (in), c2 = fgetc (out), c1 != EOF || c2 != EOF)
    {
      if (c1 != c2)
	{
	  fprintf (stderr, "Error comparing at position: %ld\n", ftell (in));
	  return 1;
	}
    }
  if (verbose)
    printf ("Comparision passed!\n");
  return 0;
}
