#ifndef IMAGECONVOLUTIONPARALLELCONSTANTMEMORY
#define IMAGECONVOLUTIONPARALLELCONSTANTMEMORY
#define KERNEL_DIMENSION 3
#define BLOCK_WIDTH 3

void applyKernelToImageParallelConstantMemory(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath);
float applyKernelPerPixelConstantMemory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelConstantMemory(float *d_image, float *d_sumArray);

__constant__ float kernelConstant[KERNEL_DIMENSION * KERNEL_DIMENSION];
__constant__ int imageWidthConstant;
__constant__ int imageHeightConstant;
__constant__ int kernelDimensionXConstant;
__constant__ int kernelDimensionYConstant;

void imageConvolutionParallelConstantMemory(const char *imageFilename, char **argv)
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
    for (int i = 0; i < 10; i++)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < numKernels; i++)
        {
            applyKernelToImageParallelConstantMemory(hData, width, height, kernels[i], kernelDimension, imagePath);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time Constant Implementation: %f \n", milliseconds);
    }
}

void applyKernelToImageParallelConstantMemory(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath)
{
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
    cudaMalloc((void **)&d_kernel, KERNEL_DIMENSION * KERNEL_DIMENSION * sizeFloat);
    cudaMalloc((void **)&d_image, sizeImageArray);
    cudaMalloc((void **)&d_sumArray, sizeImageArray);

    cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

    //constants
    cudaMemcpyToSymbol(kernelConstant, kernel, sizeof(float) * KERNEL_DIMENSION * KERNEL_DIMENSION);
    cudaMemcpyToSymbol(imageWidthConstant, &imageWidth, sizeInt);
    cudaMemcpyToSymbol(imageHeightConstant, &imageHeight, sizeInt);
    cudaMemcpyToSymbol(kernelDimensionXConstant, &kernelDimension, sizeInt);
    cudaMemcpyToSymbol(kernelDimensionYConstant, &kernelDimension, sizeInt);

    int width = imageWidth * imageHeight;
    int numBlocks = (imageWidth) / BLOCK_WIDTH;
    if (width % BLOCK_WIDTH)
        numBlocks++;

    dim3 dimGrid(numBlocks, numBlocks, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    applyKernelPerPixelParallelConstantMemory<<<dimGrid, dimBlock>>>(d_image, d_sumArray);
    cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_constant_memory_parallel_out.pgm");
    sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
__global__ void applyKernelPerPixelParallelConstantMemory(float *d_image, float *d_sumArray)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetX = (kernelDimensionXConstant - 1) / 2;
    int offsetY = (kernelDimensionYConstant - 1) / 2;
    float sum = 0.0;
    if ((y < (imageHeightConstant)) && (x < (imageWidthConstant)))
    {
        for (int j = 0; j < kernelDimensionYConstant; j++)
        {
            //Ignore out of bounds
            if (y + j < offsetY || y + j - offsetY >= imageHeightConstant)
                continue;

            for (int i = 0; i < kernelDimensionXConstant; i++)
            {
                //Ignore out of bounds
                if (x + i < offsetX || x + i - offsetX >= imageWidthConstant)
                    continue;

                float k = kernelConstant[i + j * (kernelDimensionYConstant)];
                float imageElement = d_image[y * (imageHeightConstant) + x + i - offsetX + (imageWidthConstant) * (j - 1)];

                float value = k * imageElement;
                sum = sum + value;
            }
        }
        int imageIndex = y * (imageHeightConstant) + x;

        if (sum < 0)
            sum = 0;
        else if (sum > 1)
            sum = 1;

        d_sumArray[y * (imageHeightConstant) + x] = sum;
    }
}
#endif
