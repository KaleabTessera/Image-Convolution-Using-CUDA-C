#ifndef IMAGECONVOLUTIONPARALLELTEXTUREMEMORY
#define IMAGECONVOLUTIONPARALLELTEXTUREMEMORY
#define BLOCK_WIDTH 3
#define KERNELDIMENSION 3
void applyKernelToImageParallelTextureMomory(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath);
float applyKernelPerPixelTextureMomory(int y, int x, int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);
__global__ void applyKernelPerPixelParallelTextureMomory(int *kernelX, int *kernelY, int *imageWidth, int *imageHeight, float *kernel, float *image, float *sumArray);

//2d texref
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;
texture<float, 1, cudaReadModeElementType> texRef1d;

void imageConvolutionParallelTextureMomory(const char *imageFilename, char **argv)
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
    // int kernelDimension = 3;

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
            applyKernelToImageParallelTextureMomory(hData, width, height, kernels[i], KERNELDIMENSION, imagePath);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Time Texture memory Parallel Implementation: %f \n", milliseconds);
    }
}

void applyKernelToImageParallelTextureMomory(float *image, int imageWidth, int imageHeight, float *kernel, int kernelDimension, char *imagePath)
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
    cudaMalloc((void **)&d_kernel, 9 * sizeFloat);
    cudaMalloc((void **)&d_image, sizeImageArray);
    cudaMalloc((void **)&d_sumArray, sizeImageArray);

    cudaMemcpy(d_kernelDimensionX, &kernelDimension, sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernelDimensionY, &kernelDimension, sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imageWidth, &imageWidth, sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_imageHeight, &imageHeight, sizeInt, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, 9 * sizeFloat, cudaMemcpyHostToDevice);
    cudaMemcpy(d_image, image, sizeImageArray, cudaMemcpyHostToDevice);

    //Texture memory - 2d attempt
    // cudaArray *cuArray;
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0,
    //                                                           cudaChannelFormatKindFloat);

    // cudaMallocArray(&cuArray, &channelDesc, imageWidth, imageHeight);
    // cudaMemcpyToArray(cuArray, 0, 0, image, sizeImageArray, cudaMemcpyHostToDevice);

    // // texRef1d.addressMode[0] = cudaAddressModeWrap;
    // // texRef1d.addressMode[1] = cudaAddressModeWrap;
    // // texRef1d.filterMode = cudaFilterModeLinear;
    // // texRef1d.normalized = true;

    // cudaBindTextureToArray(texRef, cuArray, channelDesc);

    cudaBindTexture(0, texRef1d, d_image, sizeImageArray);

    int width = imageWidth * imageHeight;
    int numBlocks = (imageWidth) / BLOCK_WIDTH;
    if (width % BLOCK_WIDTH)
        numBlocks++;
    dim3 dimGrid(numBlocks, numBlocks, 1);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    applyKernelPerPixelParallelTextureMomory<<<dimGrid, dimBlock>>>(d_kernelDimensionX, d_kernelDimensionY, d_imageWidth, d_imageHeight, d_kernel, d_image, d_sumArray);
    cudaMemcpy(sumArray, d_sumArray, sizeImageArray, cudaMemcpyDeviceToHost);

    //printImage(sumArray,imageWidth,imageHeight,"newImageP.txt");
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "texture_memory_parallel_out.pgm");
    sdkSavePGM(outputFilename, sumArray, imageWidth, imageHeight);
}
__global__ void applyKernelPerPixelParallelTextureMomory(int *d_kernelDimensionX, int *d_kernelDimensionY, int *d_imageWidth, int *d_imageHeight, float *d_kernel, float *d_image, float *d_sumArray)
{

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int offsetX = (*d_kernelDimensionX - 1) / 2;
    int offsetY = (*d_kernelDimensionY - 1) / 2;
    float sum = 0.0;
    if ((y < (*d_imageWidth)) && (x < (*d_imageWidth)))
    {
        for (int j = 0; j < *d_kernelDimensionY; j++)
        {
            //Ignore out of bounds
            if (y + j < offsetY || y + j - offsetY >= *d_imageHeight)
                continue;

            for (int i = 0; i < *d_kernelDimensionX; i++)
            {
                //Ignore out of bounds
                if (x + i < offsetX || x + i - offsetX >= *d_imageWidth)
                    continue;

                float k = d_kernel[i + j * (*d_kernelDimensionY)];
                float imageElement = tex1Dfetch(texRef1d, y * (*d_imageWidth) + x + i - offsetX + (*d_imageWidth) * (j - 1));

                //2d aproach no longer used
                // unsigned int xT = x + i - offsetX + (*d_imageWidth) * (j - 1);
                // unsigned int yT = y;
                // float u = xT / (float)(*d_imageWidth);
                // float v = yT / (float)(*d_imageHeight);
                // // Transform coordinates
                // u -= 0.5f;
                // v -= 0.5f;
                // float tu = xT + 0.5f;
                // float tv = yT + 0.5f;
                //  float imageElement = tex2D(texRef,tu, tv);
                float value = k * imageElement;
                sum = sum + value;
            }
        }
        int imageIndex = y * (*d_imageWidth) + x;
        if (sum < 0)
            sum = 0;
        else if (sum > 1)
            sum = 1;
        d_sumArray[y * (*d_imageWidth) + x] = sum;
    }
}
#endif
