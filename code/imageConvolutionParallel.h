#ifndef IMAGECONVOLUTIONPARALLEL
#define IMAGECONVOLUTIONPARALLEL

void applyKernelToImageParallelNaive(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath);
float applyKernelPerPixel(int y, int x,int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallel(int * kernelX, int * kernelY, int * imageWidth, int * imageHeight, float * kernel, float * image,float * sumArray);
void imageConvolutionParallel(const char *imageFilename,char **argv)
{
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
    int kernelDimension = 3;
    
    float **kernels= (float**)malloc(sizeof(float*)*numKernels);
    for(int i =0; i < numKernels;i++ ){
      kernels[i] =  (float*)malloc(sizeof(float)*100);
    }
    loadAllKernels(kernels,fp);
    fclose(fp);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    //Flip kernels to match convolution property and apply kernels to image
    for(int i =0; i < numKernels;i++ ){
      applyKernelToImageParallelNaive(hData, width, height,kernels[i],kernelDimension,imagePath);
    } 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Naive Parallel Implementation: %f \n",milliseconds);
}

void applyKernelToImageParallelNaive(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath){
  int *d_kernelDimensionX,*d_kernelDimensionY,*d_imageWidth,*d_imageHeight;
  float *d_kernel,*d_image,*d_sumArray;
  
  
  int sizeInt = sizeof(int);
  int sizeFloat = sizeof(float);
  int sizeImageArray = imageWidth * imageHeight * sizeFloat;
  float *sumArray = (float*)malloc(sizeImageArray);

  cudaMalloc((void **)&d_kernelDimensionX,sizeInt);
  cudaMalloc((void **)&d_kernelDimensionY,sizeInt);
  cudaMalloc((void **)&d_imageWidth,sizeInt);
  cudaMalloc((void **)&d_imageHeight,sizeInt);
  cudaMalloc((void **)&d_kernel,9 *sizeFloat);
  cudaMalloc((void **)&d_image, sizeImageArray);
  cudaMalloc((void **)&d_sumArray,sizeImageArray);

  cudaMemcpy(d_kernelDimensionX,&kernelDimension,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernelDimensionY,&kernelDimension,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageWidth,&imageWidth,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageHeight,&imageHeight,sizeInt,cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel,kernel,9 *sizeFloat,cudaMemcpyHostToDevice);
  cudaMemcpy(d_image,image,sizeImageArray,cudaMemcpyHostToDevice);

  dim3 gridNumber( imageHeight,imageWidth );
  dim3 threadsPerBlock( 512);
  int BLOCK_WIDTH = 3;
  int width = imageWidth*imageHeight;
  int numBlocks=(imageWidth)/BLOCK_WIDTH;
  if(width % BLOCK_WIDTH) numBlocks++;
  dim3 dimGrid(numBlocks, numBlocks, 1);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);  
  applyKernelPerPixelParallel<<<dimGrid,dimBlock>>>(d_kernelDimensionX,d_kernelDimensionY,d_imageWidth,d_imageHeight, d_kernel,d_image,d_sumArray);
  cudaMemcpy(sumArray,d_sumArray,sizeImageArray,cudaMemcpyDeviceToHost);

  //printImage(d_sumArray,imageWidth,imageHeight,"newImage.txt");
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_out.pgm");
  sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
__global__ void applyKernelPerPixelParallel(int * d_kernelDimensionX, int * d_kernelDimensionY, int * d_imageWidth, int * d_imageHeight, float * d_kernel, float * d_image,float * d_sumArray){
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;
//   printf("offsetX: %d \n",offsetX); 
  float sum =0.0;
  if ((y < (*d_imageWidth)) && (x < (*d_imageWidth))){
  for (int j = 0; j < *d_kernelDimensionY; j++) {
    //Ignore out of bounds
    if (y + j < offsetY
            || y + j - offsetY >= *d_imageHeight)
            {
                //printf("y: %d \n",y);
                continue;
            }
            

       for (int i = 0; i < *d_kernelDimensionX; i++) {
         //Ignore out of bounds
         if (x + i < offsetX
                    || x + i - offsetX >= *d_imageWidth)
                    {
                        //printf("x: %d \n",y);
                continue;
                    }

         float k = d_kernel[i + j * (*d_kernelDimensionY)];
         float imageElement =  d_image[y* (*d_imageWidth)+x + i - offsetX + (*d_imageWidth)*(j-1)];
         float value = k * imageElement;
         sum = sum +value; 
       }  
      }
      int index = y*(*d_imageWidth)+x;
      d_sumArray[y*(*d_imageWidth)+x] = sum/9;
  }
}
#endif
