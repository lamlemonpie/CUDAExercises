#import <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#import <time.h>

#define RAND_MAX = 2147483647

__global__ 
void vecAddKernel(float *d_vec1, float *d_vec2, float *d_out, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    d_out[i] = d_vec1[i] + d_vec2[i];
  }


void vecAdd(float *h_vec1, float *h_vec2, float *h_out, int n){

  int size = n*sizeof(float);   //Cantidad de memoria.
  float *d_vec1, *d_vec2, *d_out;

  cudaMalloc( (void**)&d_vec1 ,size);
  cudaMalloc( (void**)&d_vec2 ,size);
  cudaMalloc( (void**)&d_out  ,size);

  cudaMemcpy(d_vec1, h_vec1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec2, h_vec2, size, cudaMemcpyHostToDevice);


  vecAddKernel<<<ceil(n/256.0),256>>>(d_vec1,d_vec2,d_out,n);

  cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_out);

}

void genVector(float x[], int n) {
  int i;
  for (i = 0; i < n; i++)
     x[i] = random()/((double) RAND_MAX);
}

void printVector(char* title, float y[], double n) {
  printf("%s\n", title);
  for (int i = 0; i < n; i++)
     printf("%4.1f ", y[i]);
  printf("\n");
} 

int main(int argc, char **argv){
  
  int n = 30;
  float *h_vec1 = NULL;
  float *h_vec2 = NULL;
  float *h_out  = NULL;

  h_vec1 = malloc( n*sizeof(float) );
  h_vec2 = malloc( n*sizeof(float) );
  h_out  = malloc( n*sizeof(float) );
  
  string vec1,vec2,out;
  vec1 = "Vector1:";
  vec2 = "Vector2:";
  out = "Salida:";

  genVector(&h_vec1,n); genVector(&h_vec2,n);
  printVector(vec1,h_vec1,n);
  printVector(vec2,h_vec2,n);

  vecAdd(&h_vec1,&h_vec2,&h_out,n);

  printVector(out,h_out,n);
  
  return 0;
}