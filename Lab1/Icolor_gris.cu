//#include <cuda_runtime.h>
#include "lodepng.h"
#include <vector>
#include <iostream>
#include <time.h>

using namespace std;
void getError(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cout << "Error " << cudaGetErrorString(err) << std::endl;
    }
}

__global__
void blur(unsigned char* imagen_ini, unsigned char* imagen_fin, int width, int height) {

    const unsigned int ajuste = blockIdx.x*blockDim.x + threadIdx.x;
    int x = ajuste % width; //situarnos en posición correcta
    int y = (ajuste-x)/width;
    int tam_filtro = 5;
    if(ajuste < width*height) {

        float salida_ch_rojo = 0;
        float salida_ch_verde = 0;
        float salida_ch_azul = 0;
        int cantidad = 0;
        for(int ox = -tam_filtro; ox < tam_filtro+1; ++ox) {
            for(int oy = -tam_filtro; oy < tam_filtro+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int ajuste_actual = (ajuste+ox+oy*width)*3;
                    salida_ch_rojo += imagen_ini[ajuste_actual]; 
                    salida_ch_verde += imagen_ini[ajuste_actual+1];
                    salida_ch_azul += imagen_ini[ajuste_actual+2];
                    cantidad++;
                }
            }
        }
        imagen_fin[ajuste*3] = salida_ch_rojo/cantidad;
        imagen_fin[ajuste*3+1] = salida_ch_verde/cantidad;
        imagen_fin[ajuste*3+2] = salida_ch_azul/cantidad;
        }
}

void blurr (unsigned char* imagen_ini, unsigned char* imagen_fin, int width, int height) {
    unsigned char* cuda_input;
    unsigned char* cuda_output;
    //Guarda memoria en el  device 
    getError(cudaMalloc( (void**) &cuda_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy( cuda_input, imagen_ini, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));
 
    getError(cudaMalloc( (void**) &cuda_output, width*height*3*sizeof(unsigned char)));
    //declara que el tamaño del bloque tendra 512*1 hilos, se almacenara en una lista
    dim3 dim_bloque(512,1,1);
    //declara que en un grid o malla tendra el tamaño ceil((double)(width*height*3/dim_bloque.x)) de bloques X 1, se almacena en un array
    dim3 dim_grid((unsigned int) ceil((double)(width*height*3/dim_bloque.x)), 1, 1 );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	
    cudaEventRecord(start);
    blur<<<dim_grid, dim_bloque>>>(cuda_input, cuda_output, width, height); 
    cudaEventRecord(stop);

    getError(cudaMemcpy(imagen_fin, cuda_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));
    cudaEventSynchronize(stop);
	
    float ms = 0;
    std::cout << "--------------\nTiempo de proceso CUDA: ";
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ms/1000 <<" s\n--------------\n";

    getError(cudaFree(cuda_input));
    getError(cudaFree(cuda_output));
}

// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__ void colorToGreyscaleConversion(unsigned char* imagen_ini, unsigned char* imagen_fin, int width, int height) {
	int Col = threadIdx.x + blockIdx.x * blockDim.x;
	int Row = threadIdx.y + blockIdx.y * blockDim.y;
	if (Col < width && Row < height) {
	// get 1D coordinate for the grayscale image
	int greyOffset = Row*width + Col;
	// one can think of the RGB image having
	// CHANNEL times columns than the grayscale image
	int CHANNELS=3;
	int rgbOffset = greyOffset*CHANNELS;
	unsigned char r = imagen_ini[rgbOffset]; // red value for pixel
	unsigned char g = imagen_ini[rgbOffset + 2]; // green value for pixel
	unsigned char b = imagen_ini[rgbOffset + 3]; // blue value for pixel
	// perform the rescaling and store it
	// We multiply by floating point constants
	imagen_fin[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
	}
}

void color_grey(unsigned char* img_in , unsigned char* imagen_fin,int ancho,int alto ){
	unsigned char* cuda_input;
	unsigned char* cuda_output;

	getError(cudaMalloc( (void**) &cuda_input, ancho*alto*3*sizeof(unsigned char)));
	getError(cudaMalloc( (void**) &cuda_output, ancho*alto*3*sizeof(unsigned char)));
	getError(cudaMemcpy( cuda_input, img_in, ancho*alto*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));

	dim3 dimGrid(ceil(ancho/16.0), ceil(alto/16.0), 1);
   	dim3 dimBlock(16, 16, 1);
	colorToGreyscaleConversion<<<dimGrid,dimBlock>>>(cuda_input,cuda_output,ancho,alto);
	getError(cudaMemcpy(imagen_fin, cuda_output, ancho*alto*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));
}

int main() {
const char *arch_entrada = "perrito.png";
    const char *arch_salida = "perritoblur.png";
    unsigned int width, height;
    vector<unsigned char> imagen_vector;
    int error = lodepng::decode(imagen_vector, width, height, arch_entrada);
    if(error)
	cout << "Error de LODEPNG, " << error << ": " << lodepng_error_text(error) << "\n";
	unsigned char* imagen_ini = new unsigned char[(imagen_vector.size()*3)/4]; 
	unsigned char* imagen_fin = new unsigned char[(imagen_vector.size()*3)/4];
	int caracter = 0;
	    for(int i = 0; i < imagen_vector.size(); i++) {
	       if((i+1) % 4 != 0) {
		   imagen_ini[caracter] = imagen_vector.at(i);
		   //cout<<imagen_ini[caracter]<<" ";
		   imagen_fin[caracter] = 255;
		   caracter++;
	       }
	}
	vector<unsigned char> png;
std::vector<unsigned char> salida_final;	
 color_grey(imagen_ini,imagen_fin,width,height);
     for(int i = 0; i < width*height; ++i) {
         salida_final.push_back(imagen_fin[i]);
         salida_final.push_back(imagen_fin[i]);
         salida_final.push_back(imagen_fin[i]);
         salida_final.push_back(255);
         }
	//blurr(imagen_ini,imagen_fin,width,height);
    //Salida final
//    blurr(imagen_ini,imagen_fin,width,height);
//    std::vector<unsigned char> salida_final;
//    for(int i = 0; i < imagen_vector.size(); ++i) {
//        salida_final.push_back(imagen_fin[i]);
//        if((i+1) % 3 == 0) {
//            salida_final.push_back(255);
//       }
//    }

    
    
    // Guardar datos
    error = lodepng::encode(arch_salida, salida_final, width, height);

    if(error)
      cout << "Error de encoder" << error << ": "<< lodepng_error_text(error) << "\n";

	
	return 0;
}
