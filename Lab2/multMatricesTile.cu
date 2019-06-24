#include <iostream>
#include <time.h>

using namespace std;
#define TILE_WIDTH 56

__global__
void matMultKernel(float *d_M, float *d_N, float *d_P, int Width){

  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
 
  int Row = by*TILE_WIDTH + ty;
  int Col = bx*TILE_WIDTH + tx;

  float Pvalue = 0;
  int  m,k;
  for(m = 0; m < Width/TILE_WIDTH; ++m){
     Mds[ty][tx] = d_M[Row*Width+m*TILE_WIDTH + tx];
     Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty) * Width + Col];
     __syncthreads();
     for(k = 0; k < TILE_WIDTH; ++k){
      Pvalue += Mds[ty][k] * Nds[k][tx];
     }
     __syncthreads();
  }
  d_P[Row*Width + Col] = Pvalue;
}

void matMult(float* A, float* B, float* C, int n){
  int size = n*n*sizeof(float);
  float *d_A, *d_B, *d_C;

  cudaMalloc((void **) &d_A, size);
  cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_B, size);
  cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);
  cudaMalloc((void **) &d_C, size);

  dim3 dimGrid(ceil(n/1024.0),ceil(n/1024.0),1);
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
  matMultKernel<<<dimGrid, dimBlock>>>(d_A,d_B,d_C,n);
  
  cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


int main(){
  int n,i,j;
  float *h_A,*h_B,*h_C;
  n = 10000;
  /******************Time Variables*********************************/
  //clock_t time;
  cout<<"El Tam de la matriz Cuadrada es: "<<n<<" X "<<n<<endl;
  h_A = new float[n*n];
  h_B = new float[n*n];
  h_C = new float[n*n];

  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      h_A[i*n+j] = 1;
  }
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++)
      h_B[i*n+j] = 1;
  }
  /*time = clock();
  matMult(h_A,h_B,h_C,n);
  time = clock() - time;
  cout<<"El Tiempo  es: "<<(((float)time)/CLOCKS_PER_SEC)<<endl;*/
  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventRecord(start,0);

  matMult(h_A,h_B,h_C,n);

  cudaEventCreate(&stop);
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start,stop);
  printf("Elapsed time : %f ms\n" ,elapsedTime);
  //cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
   

  /*for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
        cout<<h_C[i*n+j]<<" ; ";
    }
    cout<<endl;
  }
    cout<<endl;*/
  return 0;
}
