#ifndef IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define IMAGECONVOLUTIONPARALLELSHAREDMEMORY


void applyKernelToImageParallelSharedMemory(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath);
float applyKernelPerPixelSharedMemory(int y, int x,int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedMemory(int * kernelX, int * kernelY, int * imageWidth, int * imageHeight, float * kernel, float * image,float * sumArray);
void imageConvolutionParallelSharedMemory(const char *imageFilename,char **argv)
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
      applyKernelToImageParallelSharedMemory(hData, width, height,kernels[i],kernelDimension,imagePath);
    } 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Naive Parallel Implementation: %f \n",milliseconds);
}
#define BLOCK_SIZE  8
void applyKernelToImageParallelSharedMemory(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath){
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

  
  int width = imageWidth*imageHeight;
  int numBlocks=(imageWidth)/BLOCK_SIZE;

  if(width % BLOCK_SIZE) numBlocks++;
  printf("numBlocks: %d \n",numBlocks);
  dim3 dimGrid(numBlocks, numBlocks, 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);  
  applyKernelPerPixelParallelSharedMemory<<<dimGrid,dimBlock>>>(d_kernelDimensionX,d_kernelDimensionY,d_imageWidth,d_imageHeight, d_kernel,d_image,d_sumArray);
  cudaMemcpy(sumArray,d_sumArray,sizeImageArray,cudaMemcpyDeviceToHost);
  
  printf("sum: %f \n",sumArray[0]);
  printImage(sumArray,imageWidth,imageHeight,"newImage2.txt");
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_sharedMemory_parallel_out.pgm");
  sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
// // Get a matrix element
// __device__ float GetElement(const Matrix A, int row, int col)
// {
//     return A.elements[row * A.stride + col];
// }

// __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
// {
//     float imageElement =  local_imageSection[y* (*d_imageWidth)+x + i - offsetX + (*d_imageWidth)*(j-1)];
// }

__global__ void applyKernelPerPixelParallelSharedMemory(int * d_kernelDimensionX, int * d_kernelDimensionY, int * d_imageWidth, int * d_imageHeight, float * d_kernel, float * d_image,float * d_sumArray){
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;
  float sum =0.0;

  int row = threadIdx.y;
  int col = threadIdx.x;
  __shared__ float local_imageSection[BLOCK_SIZE][BLOCK_SIZE];
  int imageIndex = y * (*d_imageWidth) + x;
  // if(row == 0 && col == 0)
    //printf("index: %f \n",d_image[imageIndex]);
    local_imageSection[row][col] = d_image[imageIndex];
  __syncthreads();

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
        //  float imageElement =  d_image[y* (BLOCK_SIZE)+x + i - offsetX + (*BLOCK_SIZE)*(j-1)];
         float imageElement =  local_imageSection[row][col];
      //   if(imageIndex < 10)
        //     printf("x: %d y: %d value: %f \n",x,y,imageElement);
          float value = k * imageElement;
          sum = sum +value; 
       }  
      }
      __syncthreads();
      //Normalising output 
      
       if(imageIndex < 10)
             printf("before sum: %f index: %d value: %f \n",sum,y*(*d_imageWidth)+x,d_sumArray[y*(*d_imageWidth)+x]);
             
      if(sum < 0)
          sum = 0;
        else if(sum >1)
          sum = 1;

      d_sumArray[y*(*d_imageWidth)+x] = sum;
      if(imageIndex < 10)
             printf("sum: %f index: %d value: %f \n",sum,y*(*d_imageWidth)+x,d_sumArray[y*(*d_imageWidth)+x]);
  }
  //}

#endif
