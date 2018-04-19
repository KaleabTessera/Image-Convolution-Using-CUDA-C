#ifndef IMAGECONVOLUTIONPARALLELSHAREDMEMORY
#define IMAGECONVOLUTIONPARALLELSHAREDMEMORY

void applyKernelToImageParallelSharedMemory(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath);
float applyKernelPerPixelSharedMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelSharedMemory(int *kernelX, int *kernelY, int *imageWidth, int *imageHeight, float *kernel, float *image, float *sumArray);
void imageConvolutionParallelSharedMemory(const char *imageFilename, char **argv)
{
  // load image from disk
  float *hData = NULL;
  unsigned int width, height;
  char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

  if (imagePath == NULL)
  {
    printf("Unable to source image file: %s\n", imageFilename);
    exit(EXIT_FAILURE);
  }

  sdkLoadPGM(imagePath, &hData, &width, &height);
  printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

  //Get Kernels
  FILE *fp = fopen("kernels.txt", "r");
  if (fp == NULL)
  {
    perror("Error in opening file");
    exit(EXIT_FAILURE);
  }
  int numKernels = getNumKernels(fp);
  int kernelDimension = 3;

  float **kernels = (float **)malloc(sizeof(float *) * numKernels);
  for (int i = 0; i < numKernels; i++)
  {
    kernels[i] = (float *)malloc(sizeof(float) * 100);
  }
  loadAllKernels(kernels, fp);
  fclose(fp);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  //Flip kernels to match convolution property and apply kernels to image
  for (int i = 0; i < numKernels; i++)
  {
    applyKernelToImageParallelSharedMemory(hData, width, height, kernels[i], kernelDimension, imagePath);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Time Naive Parallel Implementation: %f \n", milliseconds);
}

#define BLOCK_SIZE 13
void applyKernelToImageParallelSharedMemory(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath)
{
  //printImage(image, imageWidth, imageHeight, "orginalImagePartition.txt");
  int *d_kernelDimensionX, *d_kernelDimensionY, *d_imageWidth, *d_imageHeight;
  float *d_kernel, *d_image, *d_sumArray;

  int sizeInt = sizeof(int);
  int sizeFloat = sizeof(float);
  int sizeImageArray = imageWidth * imageHeight * sizeFloat;
  float *sumArray = (float *)malloc(sizeImageArray);

  cudaMalloc((void **)&d_kernelDimensionX, sizeInt);
  cudaMalloc((void **)&d_kernelDimensionY, sizeInt);
  cudaMalloc((void **)&d_imageWidth, sizeInt);
  cudaMalloc((void **)&d_imageHeight, sizeInt);
  cudaMalloc((void **)&d_kernel, 9 * sizeFloat);
  cudaMalloc((void **)&d_image, sizeImageArray);
  cudaMalloc((void **)&d_sumArray, sizeImageArray);

  cudaMemcpy(d_kernelDimensionX, &kernelDimension, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernelDimensionY, &kernelDimension, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageWidth, &imageWidth, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_imageHeight, &imageHeight, sizeInt, cudaMemcpyHostToDevice);
  cudaMemcpy(d_kernel, kernel, 9 * sizeFloat, cudaMemcpyHostToDevice);
  cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

  int width = imageWidth * imageHeight;
  int numBlocks = (imageWidth) / BLOCK_SIZE;

  if (width % BLOCK_SIZE)
    numBlocks++;
  //printf("numBlocks: %d \n", numBlocks);
  dim3 dimGrid(numBlocks, numBlocks, 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  applyKernelPerPixelParallelSharedMemory<<<dimGrid, dimBlock>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);
  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess)
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
  cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

  //printImage(sumArray, imageWidth, imageHeight, "newImage2.txt");
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_sharedMemory_parallel_out.pgm");
  sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}

__global__ void applyKernelPerPixelParallelSharedMemory(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
{
  int comp = 45;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int offsetX = (*d_kernelDimensionX - 1) / 2;
  int offsetY = (*d_kernelDimensionY - 1) / 2;

  int row = threadIdx.y;
  int col = threadIdx.x;

  __shared__ float local_imageSection[BLOCK_SIZE][BLOCK_SIZE];
  int imageIndex = y * (*d_imageWidth) + x;
  local_imageSection[row][col] = d_image[y * (*d_imageWidth) + x - 2 * blockIdx.x];

  __syncthreads();

  if ((y < (*d_imageWidth)) && (x < (*d_imageWidth))) //&& row < BLOCK_SIZE - 1 && col < BLOCK_SIZE - 1)
  {
    float sum = 0.0;
    for (int j = 0; j < *d_kernelDimensionY; j++)
    {
      //Ignore out of bounds
      if (row + j - offsetY < 0)
      {
        continue;
      }

      for (int i = 0; i < *d_kernelDimensionX; i++)
      {
        //Ignore out of bounds
        if (
            col + i - offsetX < 0)
        {
          continue;
        }
        float value;
        if (blockIdx.x > 0 && (threadIdx.x == 0))
        {
          continue;
        } // left apron

        else if ((threadIdx.x == (blockDim.x - 1)
         ||                  (threadIdx.x + j - offsetY) > blockDim.x -1 )) // right apron
        {
          continue;
        }
        else
        {
          // check col next
          // if (blockIdx.y == 0 && (threadIdx.y + i - offsetX) < 0) // top apron
          // {
          //   if (blockIdx.y * blockDim.y == 0 & blockIdx.x * blockDim.x == 0)
          //   {
          //     printf("top %d %d \n", row, col);
          //   }
          //   value = 0;
          // }
          // else if (blockIdx.y == (gridDim.y - 1) &&
          //          (threadIdx.y + i - offsetX) > blockDim.y - 1) // bottom apron
          // {
          //   if (blockIdx.y * blockDim.y == 0 & blockIdx.x * blockDim.x == 0)
          //   {
          //     printf("bottom %d %d \n", row, col);
          //   }
          //   value = 0;
          // }
          // else // safe case
          // {

          float k = d_kernel[i + j * (*d_kernelDimensionY)];
          float imageElement = local_imageSection[row + j - offsetY][col + i - offsetX];

          value = k * imageElement;
          // if ((y * (*d_imageWidth) + x  -2 * blockIdx.x) == comp)
          // {
          //   printf("block: %d thread: %d k: %f img: %f value: %f sum: %f \n",blockIdx.x,threadIdx.x, k, imageElement, value, sum);
          // }
        }
        sum = sum + value;
        // if (y * (*d_imageWidth) + x == 12)
        //   {
        //     printf("value: %f sum: %f \n",value, sum);
        //   }
      }
    }

    //Normalising output ;
    if (sum < 0)
      sum = 0;
    else if (sum > 1)
      sum = 1;
    __syncthreads();
    d_sumArray[y * (*d_imageWidth) + x - 2 * blockIdx.x] = sum;
  }
}

#endif
