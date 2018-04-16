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

  dim3 gridNumber( imageHeight,imageWidth );
  dim3 threadsPerBlock( 512);
  int BLOCK_WIDTH = 3;
  int width = imageWidth*imageHeight;
  int numBlocks=(imageWidth)/BLOCK_WIDTH;
  if(width % BLOCK_WIDTH) numBlocks++;
  dim3 dimGrid(numBlocks, numBlocks, 1);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);  
  applyKernelPerPixelParallelSharedMemory<<<dimGrid,dimBlock>>>(d_kernelDimensionX,d_kernelDimensionY,d_imageWidth,d_imageHeight, d_kernel,d_image,d_sumArray);
  cudaMemcpy(sumArray,d_sumArray,sizeImageArray,cudaMemcpyDeviceToHost);
  
  printImage(sumArray,imageWidth,imageHeight,"newImage2.txt");
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_sharedMemory_parallel_out.pgm");
  sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
__global__ void applyKernelPerPixelParallelSharedMemory(int * d_kernelDimensionX, int * d_kernelDimensionY, int * d_imageWidth, int * d_imageHeight, float * d_kernel, float * d_image,float * d_sumArray){
  int y = blockIdx.y*blockDim.y+threadIdx.y;
  int x = blockIdx.x*blockDim.x+threadIdx.x;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;
  float sum =0.0;
  
  int index = y* (*d_imageWidth)+x;
  
//   int size = (*d_imageHeight) * (*d_imageWidth);
//   int size = 9;
//   __shared__ float local_imageSection[3][3];
 __shared__ float local_imageSection[2621];
   //local_imageSection[index]=0.0;
   //__syncthreads();
  
  int i=index;
  int size = i + 9;
//   printf("Index1: %d  \n",i);
  int stride=blockIdx.x * gridDim.x;
        int imageIndex = threadIdx.y* (*d_imageWidth)+threadIdx.x + threadIdx.x;
        //   local_imageSection[threadIdx.x][threadIdx.y] = d_image[imageIndex];
        local_imageSection[y * 3 + x] = d_image[imageIndex];
         printf("Index: %d %f\n",y * 3 + x,local_imageSection[y * 3 + x]);
        // printf("test: %f \n", local_imageSection[j * 3 + i] );
           if(index == 0){
            //   printf("Index: %d %f\n",imageIndex,d_image[imageIndex]);
        //       //printf("Index: %d, %d = %f \n",i,j,local_imageSection[j * 3 + i]);
           }  
__syncthreads();
  
  if(index == 100)
           printf("test: %f * %f \n",y,threadIdx.y);

  if ((y < (*d_imageWidth)) && (x < (*d_imageWidth))){
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
        //  float imageElement =  local_imageSection[y* (*d_imageWidth)+x + i - offsetX + (*d_imageWidth)*(j-1)];
        //  printf("test: %f \n", local_imageSection[y* (3)+x + i - offsetX + (3)*(j-1)] );
          
         float imageElement =  local_imageSection[j * 3 + i];
        
         float value = k*imageElement;
        //   if(index == 0)
        //    printf("test: %f * %f \n",k,imageElement);
         sum = sum +value; 
          
       }  
      }
      //Normalising output 
      if(sum < 0)
          sum = 0;
        else if(sum >1)
          sum = 1;
    //   if(sum > 0)
        //  printf("sum: %f \n",sum);
        if(index == 0)
           printf("test:%f \n",d_sumArray[0] );

      d_sumArray[y*(*d_imageWidth)+x] = sum;
    
  }
}
#endif
