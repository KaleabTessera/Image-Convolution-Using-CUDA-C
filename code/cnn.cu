#include <iostream>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "helpers.h"
#include "imageConvolutionSerial.h"
#include "imageConvolutionParallel.h"


const char *imageFilename = "image21.pgm";

int main(int argc, char **argv)
{
  printf("Image convolution project \n");
  printf("Please select an option \n");
  printf("1 - Serial Implementation \n");
  printf("2 - Naive parallel implementation \n");
  
  int option;
  scanf("%d", &option);

  switch(option) {
    case 1  :
      imageConvolutionSerial(imageFilename,argv);
      break; 
   
    case 2  :
       imageConvolutionParallel(imageFilename,argv);
       break; 
   
    default : 
      printf("Incorrect input \n");
 }

  return 0;
}


