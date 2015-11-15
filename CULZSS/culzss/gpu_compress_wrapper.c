/***************************************************************************
*          CUDA LZSS 
*   Authors  : Adnan Ozsoy, Martin Swany,Indiana University - Bloomington
*   Date    : April 11, 2011

****************************************************************************
	Copyright 2011 Adnan Ozsoy, Martin Swany, Indiana University - Bloomington

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
	
****************************************************************************/

 
#include <stdio.h>
#include <stdlib.h>
#include "gpu_compress.h"

   

extern int compression_kernel_wrapper(unsigned char *buffer, long buf_length,unsigned char * compressed_buffer, long * comp_length, int compression_type,int wsize, int numthre);
extern void decompression_kernel_wrapper(unsigned char *buffer, long buf_length, long * comp_length, int compression_type,int wsize, int numthre);


//call kernel wrapper for cuda
int gpu_compress(unsigned char *buffer, long buf_length,unsigned char * compressed_buffer, long * comp_length, int compression_type,int wsize, int numthre){

  return compression_kernel_wrapper(buffer, buf_length, compressed_buffer, comp_length, compression_type, wsize,  numthre);

}


//call kernel wrapper for cuda
int gpu_decompress(unsigned char *buffer, long buf_length,unsigned char * decompressed_buffer, long * comp_length, int compression_type,int wsize, int numthre){


  decompression_kernel_wrapper(buffer, buf_length, comp_length, compression_type, wsize,  numthre);

  return 0;
}
