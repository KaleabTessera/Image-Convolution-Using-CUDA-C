#ifndef _helper_h   
#define _helper_h

void printImage(float *image,int width,int height,char * fileName){
   FILE *f = fopen(fileName, "w");
   if (f == NULL)
   {
    printf("Error opening file!\n");
    exit(1);
   }

   for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            fprintf(f,"%f ", image[i*width+j]);
        }
        fprintf(f,"\n");
    }
}

void printKernel(float* kernel, int kernelDimension){
  for (int i = 0; i < kernelDimension; i++) {
       for (int j = 0; j < kernelDimension; j++) {
           printf("%f ", kernel[i*kernelDimension+j]);
       }
       printf("\n");
   }
}

int getNumKernels(FILE* fp){
  int ch, lines=0;
  while(!feof(fp))
  {
      ch = fgetc(fp);
      if(ch == '\n')
      {
        lines++;
      }
  }
  rewind(fp);
  return lines;
}
#endif 