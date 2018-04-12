#ifndef _helper_h   
#define _helper_h
void flipKernel(float* kernel, int kernelDimension);
void loadKernels(float * kernel, char buf[512]);
void loadAllKernels(float ** kernels,  FILE* fp);
int getNumKernels(FILE* fp);

void loadAllKernels(float ** kernels, FILE* fp){   
    char buf[512];
    int index = 0;
    
   
    while (fgets(buf, sizeof(buf), fp) != NULL)
    {
      loadKernels(kernels[index],buf);
      index++;
    }
      
}

void loadKernels(float * kernel, char buf[512]){ 
    int count = 0;
    buf[strlen(buf) - 1] = '\0';
    const char delimeter[2] = ",";
    char* token = strtok(buf, delimeter);
    while( token != NULL){
      sscanf(token, "%f,", &kernel[count] );
      token = strtok(NULL,delimeter);
      count = count + 1;
    } 
}

//https://www.geeksforgeeks.org/write-a-program-to-reverse-an-array-or-string/
void flipKernel(float* kernel, int kernelDimension){
  int temp;
  int start = 0;
  int end = kernelDimension * kernelDimension - 1;
  while (start < end)
  {
      temp = kernel[start];   
      kernel[start] = kernel[end];
      kernel[end] = temp;
      start++;
      end--;
  }   
      
}

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
    return 1;
//   int ch, lines=0;
//   while(!feof(fp))
//   {
//        printf("2");
//       ch = fgetc(fp);
//       if(ch == '\n')
//       {
//         lines++;
//       }
//   }
//   rewind(fp);
//   return lines;
}
#endif 