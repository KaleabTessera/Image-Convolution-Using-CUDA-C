#include <iostream>
#include <stdio.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "helpers.h"


const char *imageFilename = "image21.pgm";

void imageConvolution(int argc, char **argv);
void printImage(float *image,int width,int height);
void printKernel(float * kernel, int kernelDimension);
void applyKernelToImageSerial(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath);
void flipKernel(float* kernel, int kernelDimension);
void loadKernels(float * kernel, char buf[512]);
void loadAllKernels(float ** kernels,  FILE* fp);
int getNumKernels(FILE* fp);
<<<<<<< HEAD
float applyKernelPerPixel(int y, int x,int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
void applyKernelToImageParallelNaive(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath);
__global__ void applyKernelPerPixelParallel(int * kernelX, int * kernelY, int * imageWidth, int * imageHeight, float * kernel, float * image,float * sumArray);
=======
float applyKernelPerPixel(int y, int x,int kernelX, int kernelY, int imageWidth, int imageHeight, float * kernel,float *image);
void applyKernelToImageParallelNaive(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath);
>>>>>>> 7351ef51280844a84d3c1bdc98103d7670d541d2

int main(int argc, char **argv)
{
  imageConvolution(argc,argv);
  return 0;
}


void imageConvolution(int argc, char **argv)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
	// load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename,argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);
    
    //Get Kernels
    FILE* fp = fopen("kernels.txt", "r");
    if(fp == NULL) {
      perror("Error in opening file");
      exit(EXIT_FAILURE);
   }

    int numKernels = getNumKernels(fp);
    //printf("%d",numKernels);
    int kernelDimension = 3;
    
    float **kernels= (float**)malloc(sizeof(float*)*numKernels);
    for(int i =0; i < numKernels;i++ ){
      kernels[i] =  (float*)malloc(sizeof(float)*100);
    }
    loadAllKernels(kernels,fp);
    fclose(fp);
    cudaEventRecord(start);
    //Flip kernels to match convolution property and apply kernels to image
    for(int i =0; i < numKernels;i++ ){
<<<<<<< HEAD
      applyKernelToImageParallelNaive(hData, width, height,kernels[i],kernelDimension,imagePath);
=======
      applyKernelToImageSerial(hData, width, height,kernels[i],kernelDimension,imagePath);
>>>>>>> 7351ef51280844a84d3c1bdc98103d7670d541d2
    } 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f \n",milliseconds);
}

void applyKernelToImageParallelNaive(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath){
  int *d_kernelDimensionX,*d_kernelDimensionY,*d_imageWidth,*d_imageHeight;
  float *d_kernel,*d_image,*d_sumArray;
  
  float *sumArray = (float*)malloc(imageWidth*imageHeight);
  int sizeInt = sizeof(int);
  int sizeFloat = sizeof(float);

  cudaMalloc((void **)&d_kernelDimensionX,sizeInt);
  cudaMalloc((void **)&d_kernelDimensionY,sizeInt);
  cudaMalloc((void **)&d_imageWidth,sizeInt);
  cudaMalloc((void **)&d_imageHeight,sizeInt);
  cudaMalloc((void **)&d_kernel,sizeFloat);
  cudaMalloc((void **)&d_image,sizeFloat);
  cudaMalloc((void **)&d_sumArray,sizeFloat);

  cudaMemcpy(d_kernelDimensionX,&kernelDimension,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernelDimensionY,&kernelDimension,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageWidth,&imageWidth,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageHeight,&imageHeight,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel,&kernel,sizeFloat,cudaMemcpyHostToDevice);
  cudaMemcpy(d_image,&image,sizeFloat,cudaMemcpyHostToDevice);

  dim3 gridNumber( imageHeight,imageWidth );
  dim3 threadsPerBlock( 512);
  //printf("%f \n", kernel[0]); 
  applyKernelPerPixelParallel<<<8,160>>>(d_kernelDimensionX,d_kernelDimensionY,d_imageWidth,d_imageHeight, d_kernel,d_image,d_sumArray);

  cudaMemcpy(&sumArray,d_sumArray,sizeFloat,cudaMemcpyHostToDevice);
  printf("%f \n",sumArray[0]);
  unsigned int size = imageWidth * imageHeight * sizeof(float);
  float *newImage = (float *) malloc(size);

  printImage(newImage,imageWidth,imageHeight,"newImage.txt");
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, newImage, imageWidth, imageHeight);
}

void applyKernelToImageSerial(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath){
  unsigned int size = imageWidth * imageHeight * sizeof(float);
  float *newImage = (float *) malloc(size);
  for(int y =0; y < imageHeight; y++){
    for(int x=0; x < imageWidth; x++){
      float sum = applyKernelPerPixel(y,x,kernelDimension,kernelDimension,imageWidth,imageHeight, kernel,image);
      //Normalising output - image doesn't get brighter or dimmer
       newImage[y*imageWidth+x] = sum/(kernelDimension * kernelDimension);
    }
  }
  printImage(newImage,imageWidth,imageHeight,"newImage.txt");
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, newImage, imageWidth, imageHeight);
}

float applyKernelPerPixel(int y, int x,int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image){
  float sum = 0;
  int offsetX = (kernelX - 1) / 2;
  int offsetY = (kernelY - 1) / 2;

  for (int j = 0; j < kernelY; j++) {
    //Ignore out of bounds
    if (y + j < offsetY
            || y + j - offsetY >= imageHeight)
            continue;

       for (int i = 0; i < kernelX; i++) {
         //Ignore out of bounds
         if (x + i < offsetX
                    || x + i - offsetX >= imageWidth)
            continue;

         float k = kernel[i + j * kernelY];
         float imageElement =  image[y*imageWidth+x + i - offsetX + imageWidth*(j-1)];
         float value = k * imageElement;
         sum = sum +value; 
       }     
      }
      return sum;
}

__global__ void applyKernelPerPixelParallel(int * d_kernelDimensionX, int * d_kernelDimensionY, int * d_imageWidth, int * d_imageHeight, float * d_kernel, float * d_image,float * d_sumArray){
  int x= threadIdx.x;
  int y= threadIdx.y;
  //printf("%d \n",x); 
  //printf("%d \n",y); 
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;
  float sumy =0;
  for (int j = 0; j < *d_kernelDimensionY; j++) {
    //Ignore out of bounds
    if (y + j < offsetY
            || y + j - offsetY >= *d_imageHeight)
            continue;

       for (int i = 0; i < *d_kernelDimensionX; i++) {
         //Ignore out of bounds
         if (x + i < offsetX
                    || x + i - offsetX >= *d_imageWidth)
            continue;

         float k = d_kernel[i + j * (*d_kernelDimensionY)];
        //  printf("%d \n",(i + j * (*d_kernelDimensionY))); 
         printf("%f \n", d_kernel[0]); 
         float imageElement =  d_image[y* (*d_imageWidth)+x + i - offsetX + (*d_imageWidth)*(j-1)];
         float value = k * imageElement;
         sumy = sumy +value; 
        
       } 
       //printf("%f \n",sumy);    
      }
      
      d_sumArray[y+x] = sumy;
      //return sum;
}
