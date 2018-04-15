#ifndef IMAGECONVOLUTIONSERIAL
#define IMAGECONVOLUTIONSERIAL

void applyKernelToImageSerial(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath);
void flipKernel(float* kernel, int kernelDimension);
void loadKernels(float * kernel, char buf[512]);
void loadAllKernels(float ** kernels,  FILE* fp);
int getNumKernels(FILE* fp);
float applyKernelPerPixel(int y, int x,int kernelX, int kernelY, int imageWidth, int imageHeight, float *kernel, float *image);

void imageConvolutionSerial(const char *imageFilename,char **argv)
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
      applyKernelToImageSerial(hData, width, height,kernels[i],kernelDimension,imagePath);
    } 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time Serial Implementation: %f \n",milliseconds);
}
void applyKernelToImageSerial(float * image, int imageWidth, int imageHeight, float * kernel, int kernelDimension, char *imagePath){
  printImage(image,imageWidth,imageHeight,"originalImage.txt");
  unsigned int size = imageWidth * imageHeight * sizeof(float);
  float *newImage = (float *) malloc(size);
  for(int y =0; y < imageHeight; y++){
    for(int x=0; x < imageWidth; x++){
      float sum = applyKernelPerPixel(y,x,kernelDimension,kernelDimension,imageWidth,imageHeight, kernel,image);
      //Normalising output 
        if(sum < 0)
          sum = 0;
        else if(sum >1)
          sum = 1;
       newImage[y*imageWidth+x] = sum;
    }
  }
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 4, "_serial_out.pgm");
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

#endif
