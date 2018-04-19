#include <iostream>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "helpers.h"
#include "imageConvolutionSerial.h"
#include "imageConvolutionParallel.h"
#include "imageConvolutionParallelSharedMemory.h"
#include "imageConvolutionParallelConstantMemory.h"
#include "imageConvolutionTextureMemory.h"


// const char *imageFilename = "lena_bw.pgm";
const char *imageFilename = "galaxy.ascii.pgm";
#define KERNELDIMENSION 3 

int main(int argc, char **argv)
{
  printf("Image convolution project \n");
  printf("Please select an option \n");
  printf("1 - Serial Implementation \n");
  printf("2 - Naive parallel implementation \n");
  printf("3 - Shared memory implementation \n");
  printf("4 - Constant memory implementation \n");
  printf("5 - Texture memory implementation \n ");  
  int option;
  scanf("%d", &option);

  switch(option) {
    case 1  :
      imageConvolutionSerial(imageFilename,argv);
      break; 
   
    case 2  :
       imageConvolutionParallel(imageFilename,argv);
       break; 

    case 3  :
      imageConvolutionParallelSharedMemory(imageFilename,argv);
      break;

    case 4:
	imageConvolutionParallelConstantMemory(imageFilename,argv);
	break; 
    
    case 5:
	imageConvolutionParallelTextureMomory(imageFilename,argv);
	break;

    default : 
      printf("Incorrect input \n");
 }

  return 0;
}


