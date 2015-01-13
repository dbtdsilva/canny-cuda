
// Based on CUDA SDK template from NVIDIA

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>

// includes, project
#include <cutil_inline.h>

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))

#define MAX_BRIGHTNESS 255
 
// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;

// Device constant
__constant__ int const_nx;
__constant__ int const_ny;
__constant__ int const_khalf;

// convolution of in image to out image using kernel of kn width
void convolution(const pixel_t *in, pixel_t *out, const float *kernel,
                 const int nx, const int ny, const int kn)
{
    assert(kn % 2 == 1);
    assert(nx > kn && ny > kn);
    const int khalf = kn / 2;
 
    for (int m = khalf; m < nx - khalf; m++)
        for (int n = khalf; n < ny - khalf; n++) {
            float pixel = 0.0;
            size_t c = 0;
            for (int j = -khalf; j <= khalf; j++)
                for (int i = -khalf; i <= khalf; i++) {
                    pixel += in[(n + j) * nx + m + i] * kernel[c];
                    c++;
                }
 
            out[n * nx + m] = (pixel_t)pixel;
        }
}

// determines min and max of in image
void min_max(const pixel_t *in, const int nx, const int ny, pixel_t *pmin, pixel_t *pmax)
{
    int min = INT_MAX, max = -INT_MAX;
 
        for (int m = 0; m < nx; m++)
            for (int n = 0; n < ny ; n++) {
                int pixel = in[n*nx + m];
                if (pixel < min)
                    min = pixel;
                if (pixel > max)
                    max = pixel;
                }
    *pmin = min; *pmax = max;
}
 
// normalizes inout image using min and max values
void normalize(  pixel_t *inout, 
                 const int nx, const int ny, const int kn,
                 const int min, const int max)
{
    const int khalf = kn / 2;

    for (int m = khalf; m < nx - khalf; m++)
        for (int n = khalf; n < ny - khalf; n++) {
 
            pixel_t pixel = MAX_BRIGHTNESS * ((int)inout[n * nx + m] -(float) min) / ((float)max - (float)min);
            inout[n * nx + m] = pixel;
        }
}

 
/*
 * gaussianFilter:
 * http://www.songho.ca/dsp/cannyedge/cannyedge.html
 * determine size of kernel (odd #)
 * 0.0 <= sigma < 0.5 : 3
 * 0.5 <= sigma < 1.0 : 5
 * 1.0 <= sigma < 1.5 : 7
 * 1.5 <= sigma < 2.0 : 9
 * 2.0 <= sigma < 2.5 : 11
 * 2.5 <= sigma < 3.0 : 13 ...
 * kernelSize = 2 * int(2*sigma) + 3;
 */
void gaussian_filter(const pixel_t *in, pixel_t *out,
                     const int nx, const int ny, const float sigma)
{
    const int n = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(n / 2.0);
    float kernel[n * n]; // variable length array
 
    fprintf(stderr, "gaussian_filter: kernel size %d, sigma=%g\n",
            n, sigma);
    size_t c = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            kernel[c] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) +
                                    pow((j - mean) / sigma, 2.0)))
                        / (2 * M_PI * sigma * sigma);
            c++;
        }
 
    convolution(in, out, kernel, nx, ny, n);
    pixel_t max, min;
    min_max(out, nx, ny, &min, &max);
    normalize(out, nx, ny, n, min, max);
}

// Canny non-maximum suppression
void non_maximum_supression(const pixel_t *after_Gx, const pixel_t * after_Gy, const pixel_t *G, pixel_t *nms, 
                            const int nx, const int ny)
{
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            const int nn = c - nx;
            const int ss = c + nx;
            const int ww = c + 1;
            const int ee = c - 1;
            const int nw = nn + 1;
            const int ne = nn - 1;
            const int sw = ss + 1;
            const int se = ss - 1;
 
            const float dir = (float)(fmod(atan2(after_Gy[c],
                                                 after_Gx[c]) + M_PI,
                                           M_PI) / M_PI) * 8;
 
            if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
                 G[c] > G[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && G[c] > G[nw] &&
                 G[c] > G[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && G[c] > G[nn] &&
                 G[c] > G[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && G[c] > G[ne] &&
                 G[c] > G[sw]))   // 135 deg
                nms[c] = G[c];
            else
                nms[c] = 0;
        }
}

// edges found in first pass for nms > tmax
void first_edges(const pixel_t *nms, pixel_t *reference, 
                 const int nx, const int ny, const int tmax)
{
 
    size_t c = 1;
    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            if (nms[c] >= tmax) { // trace edges
                reference[c] = MAX_BRIGHTNESS;
            }
            c++;
        }
        c+=2; // because borders are not considered
    }
}

 
// edges found in after first passes for nms > tmin && neighbor is edge
void hysteresis_edges(const pixel_t *nms, pixel_t *reference, 
                      const int nx, const int ny, const int tmin, bool *pchanged)
{
    // Tracing edges with hysteresis . Non-recursive implementation.
    for (int i = 1; i < nx - 1; i++) {
        for (int j = 1; j < ny - 1; j++) {
                    size_t t = i + j * nx;

                    int nbs[8]; // neighbours
                    nbs[0] = t - nx;     // nn
                    nbs[1] = t + nx;     // ss
                    nbs[2] = t + 1;      // ww
                    nbs[3] = t - 1;      // ee
                    nbs[4] = nbs[0] + 1; // nw
                    nbs[5] = nbs[0] - 1; // ne
                    nbs[6] = nbs[1] + 1; // sw
                    nbs[7] = nbs[1] - 1; // se
 
                    if (nms[t] >= tmin && reference[t] == 0) {
                       for(int k = 0; k < 8; k++) 
                           if (reference[nbs[k]] != 0) {
                               reference[t] = MAX_BRIGHTNESS;
                               *pchanged = true;
                           }
                    }
            }
        }
}

/*
 * Links:
 * http://en.wikipedia.org/wiki/Canny_edge_detector
 * http://www.tomgibara.com/computer-vision/CannyEdgeDetector.java
 * http://fourier.eng.hmc.edu/e161/lectures/canny/node1.html
 * http://www.songho.ca/dsp/cannyedge/cannyedge.html
 *
 * Note: T1 and T2 are lower and upper thresholds.
 */

//canny edge detector code to run on the host
void cannyHost( const int *h_idata, const int w, const int h, 
                const int tmin,            // tmin canny parameter
                const int tmax,            // tmax canny parameter
                const float sigma,         // sigma canny parameter
                int * reference)
{
    const int nx = w;
    const int ny = h;
 
    pixel_t *G        = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
    pixel_t *after_Gx = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
    pixel_t *after_Gy = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
    pixel_t *nms      = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
 
    if (G == NULL || after_Gx == NULL || after_Gy == NULL ||
        nms == NULL || reference == NULL) {
        fprintf(stderr, "canny_edge_detection:"
                " Failed memory allocation(s).\n");
        exit(1);
    }
 
    // Gaussian filter
    gaussian_filter(h_idata, reference, nx, ny, sigma);
 
    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
 
    // Gradient along x
    convolution(reference, after_Gx, Gx, nx, ny, 3);
 
    const float Gy[] = { 1, 2, 1,
                         0, 0, 0,
                        -1,-2,-1};
 
    // Gradient along y
    convolution(reference, after_Gy, Gy, nx, ny, 3);
 
    // Merging gradients
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)( after_Gy[c]) ));
        }
 
    // Non-maximum suppression, straightforward implementation.
    non_maximum_supression(after_Gx, after_Gy, G, nms, nx, ny);

    // edges with nms >= tmax
    memset(reference, 0, sizeof(pixel_t) * nx * ny);
    first_edges(nms, reference, nx, ny, tmax);

    // edges with nms >= tmin && neighbor is edge
    bool changed;
    do {
        changed = false;
        hysteresis_edges(nms, reference, nx, ny, tmin, &changed);
    } while (changed==true);
 
    free(after_Gx);
    free(after_Gy);
    free(G);
    free(nms);
}   

/* DEVICE OPERATIONS */

__global__  void convolution_kernel(const pixel_t *in, const float *kernel, pixel_t *out) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x + const_khalf;
    int y = threadIdx.y + blockIdx.y * blockDim.y + const_khalf;

    if((x < (const_nx - const_khalf)) && (y < (const_ny - const_khalf)))
    {
        const int width = 18;
        const int height = 34;
        const int size = width * height;
        const bool vLimit = (y == const_ny-2);
        const bool hLimit = (x == const_nx-2);
        __shared__ pixel_t subMatrix[size];

        int sub_x = threadIdx.x + const_khalf;
        int sub_y = threadIdx.y + const_khalf;

        if(sub_x == 1 && sub_y == 1)
            subMatrix[(sub_y-1)*width + sub_x-1] = in[(y-1)*const_nx + x-1];
        else if(sub_x == 1 && (vLimit || sub_y == height-2))
            subMatrix[(sub_y+1)*width + sub_x-1] = in[(y+1)*const_nx + x-1];
        else if((hLimit || sub_x == width-2) && sub_y == 1)
            subMatrix[(sub_y-1)*width + sub_x+1] = in[(y-1)*const_nx + x+1];
        else if((hLimit || sub_x == width-2) && (vLimit || sub_y == height-2))
            subMatrix[(sub_y+1)*width + sub_x+1] = in[(y+1)*const_nx + x+1];

        if(sub_x == 1)
            subMatrix[sub_y*width + sub_x-1] = in[y*const_nx + x-1];
        else if(hLimit || sub_x == width-2)
            subMatrix[sub_y*width + sub_x+1] = in[y*const_nx + x+1];

        if(sub_y == 1)
            subMatrix[(sub_y-1)*width + sub_x] = in[(y-1)*const_nx + x];
        else if(vLimit || sub_y == height-2)
            subMatrix[(sub_y+1)*width + sub_x] = in[(y+1)*const_nx + x];

        subMatrix[sub_y*width + sub_x] = in[y*const_nx + x];

        __syncthreads();

        float pixel = 0.0;
        size_t c = 0;
        for(int j = -const_khalf; j <= const_khalf; j++) 
            for(int i = -const_khalf; i <= const_khalf; i++)
                pixel += subMatrix[(sub_y+j)*width + sub_x+i] * kernel[c++];
        out[y*const_nx + x] = (pixel_t) pixel;
    }
}

// convolution of in image to out image using kernel of kn width
void convolution_device(const pixel_t *in, pixel_t *out, const float *kernel,
                 const int nx, const int ny, const int kn)
{
    assert(kn % 2 == 1);
    assert(nx > kn && ny > kn);
    
    const int khalf = kn / 2;
    
    cudaMemcpyToSymbol(const_khalf, &khalf, sizeof(int));

    dim3 gridSize(ceil((nx - 2*khalf)/ 16.0), ceil((ny - 2*khalf)/ 32.0));              
    dim3 blockSize(16, 32);             // 512 threads (x - 16, y - 32)
    
    convolution_kernel <<<gridSize, blockSize>>> (in, kernel, out);
}

__global__  void non_maximum_supression_kernel(const pixel_t *afterGx, const pixel_t *afterGy,
                            const pixel_t *G, pixel_t *nms)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int y = threadIdx.y + blockIdx.y * blockDim.y + 1;
    
    if((x < (const_nx - 1)) && (y < (const_ny - 1)))
    {
        const int width = 18;
        const int height = 34;
        const int size = width * height;
        const bool vLimit = (y == const_ny-2);
        const bool hLimit = (x == const_nx-2);
        __shared__ pixel_t subMatrix[size];

        int sub_x = threadIdx.x + 1;
        int sub_y = threadIdx.y + 1;

        if(sub_x == 1 && sub_y == 1)
            subMatrix[(sub_y-1)*width + sub_x-1] = G[(y-1)*const_nx + x-1];
        else if(sub_x == 1 && (vLimit || sub_y == height-2))
            subMatrix[(sub_y+1)*width + sub_x-1] = G[(y+1)*const_nx + x-1];
        else if((hLimit || sub_x == width-2) && sub_y == 1)
            subMatrix[(sub_y-1)*width + sub_x+1] = G[(y-1)*const_nx + x+1];
        else if((hLimit || sub_x == width-2) && (vLimit || sub_y == height-2))
            subMatrix[(sub_y+1)*width + sub_x+1] = G[(y+1)*const_nx + x+1];

        if(sub_x == 1)
            subMatrix[sub_y*width + sub_x-1] = G[y*const_nx + x-1];
        else if(hLimit || sub_x == width-2)
            subMatrix[sub_y*width + sub_x+1] = G[y*const_nx + x+1];

        if(sub_y == 1)
            subMatrix[(sub_y-1)*width + sub_x] = G[(y-1)*const_nx + x];
        else if(vLimit || sub_y == height-2)
            subMatrix[(sub_y+1)*width + sub_x] = G[(y+1)*const_nx + x];

        subMatrix[sub_y*width + sub_x] = G[y*const_nx + x];

        int c = y*const_nx + x;
        int sub_c = sub_x + width*sub_y;
        int nn = c - width;
        int ss = c + width;
        int ww = c + 1;
        int ee = c - 1;
        int nw = nn + 1;
        int ne = nn - 1;
        int sw = ss + 1;
        int se = ss - 1;

        __syncthreads();

        float dir = (float) (fmod(atan2((double) afterGy[c],(double) afterGx[c]) + M_PI, M_PI) / M_PI) * 8;

        if(((dir <= 1 || dir > 7) && subMatrix[sub_c] > subMatrix[ee] && subMatrix[sub_c] > subMatrix[ww]) ||
           ((dir > 1 && dir <= 3) && subMatrix[sub_c] > subMatrix[nw] && subMatrix[sub_c] > subMatrix[se]) ||
           ((dir > 3 && dir <= 5) && subMatrix[sub_c] > subMatrix[nn] && subMatrix[sub_c] > subMatrix[ss]) ||
           ((dir > 5 && dir <= 7) && subMatrix[sub_c] > subMatrix[ne] && subMatrix[sub_c] > subMatrix[sw]))
            nms[c] = subMatrix[sub_c];
        else
            nms[c] = 0;
    }
}

// Canny non-maximum suppression
void non_maximum_supression_device(const pixel_t *after_Gx, const pixel_t * after_Gy,
                    const pixel_t *G, pixel_t *nms, const int nx, const int ny)
{
    dim3 gridSize(ceil((nx - 2)/ 16.0), ceil((ny - 2)/ 32.0));              
    dim3 blockSize(16, 32);             // 512 threads (x - 16, y - 32)
    
    non_maximum_supression_kernel <<<gridSize, blockSize>>> (after_Gx, after_Gy, G, nms);
}

__global__ void merging_gradients_kernel(const pixel_t *afterGx, const pixel_t *afterGy,
                                        pixel_t *G)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int y = threadIdx.y + blockIdx.y * blockDim.y + 1;
    
    if((x < (const_nx - 1)) && (y < (const_ny - 1)))
    {
        const int c = x + const_nx * y;
        G[c] = (pixel_t)(hypot((double)(afterGx[c]), (double)( afterGy[c]) ));
    }
}

void merging_gradients_device(const pixel_t *after_Gx, const pixel_t *after_Gy, pixel_t *G,
                            const int nx, const int ny)
{
    dim3 gridSize(ceil((nx - 2)/ 16.0), ceil((ny - 2)/ 32.0));              
    dim3 blockSize(16, 32);             // 512 threads (x - 16, y - 32)
    
    merging_gradients_kernel <<<gridSize, blockSize>>> (after_Gx, after_Gy, G);
}

// canny edge detector code to run on the GPU
void cannyDevice( const int *h_idata, const int w, const int h, 
                  const int tmin, const int tmax, 
                  const float sigma,
                  int * h_odata)
{
    const int nx = w;
    const int ny = h;
    const int memSize = nx * ny * sizeof(pixel_t);
    const int gradSize = 3 * 3 * sizeof(float);

    cudaMemcpyToSymbol(const_nx, &nx, sizeof(int));
    cudaMemcpyToSymbol(const_ny, &ny, sizeof(int));
 
    pixel_t *nms      = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));

    // cuda pointers
    pixel_t *dev_h_odata;
    pixel_t *dev_G;
    pixel_t *dev_after_Gx;
    pixel_t *dev_after_Gy;
    pixel_t *dev_nms;
    float *dev_grad;

    cudaMalloc((void**) &dev_h_odata, memSize);
    cudaMalloc((void**) &dev_G, memSize);
    cudaMalloc((void**) &dev_after_Gx, memSize);
    cudaMalloc((void**) &dev_after_Gy, memSize);
    cudaMalloc((void**) &dev_nms, memSize);
    cudaMalloc((void**) &dev_grad, gradSize);

    cudaMemset(dev_h_odata, 0, memSize);
    cudaMemset(dev_G, 0, memSize);
    cudaMemset(dev_after_Gx, 0, memSize);
    cudaMemset(dev_after_Gy, 0, memSize);
    cudaMemset(dev_nms, 0, memSize);

    if (nms == NULL || h_odata == NULL) {
        fprintf(stderr, "canny_edge_detection:"
                " Failed memory allocation(s).\n");
        exit(1);
    }
 
    // Gaussian filter using convolution_device
    gaussian_filter(h_idata, h_odata, nx, ny, sigma);
    
    cudaMemcpy(dev_h_odata, h_odata, memSize, cudaMemcpyHostToDevice);

    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
 
    cudaMemcpy(dev_grad, Gx, gradSize, cudaMemcpyHostToDevice);

    // Gradient along x
    convolution_device(dev_h_odata, dev_after_Gx, dev_grad, nx, ny, 3);
 
    const float Gy[] = { 1, 2, 1,
                         0, 0, 0,
                        -1,-2,-1};
 
    cudaMemcpy(dev_grad, Gy, gradSize, cudaMemcpyHostToDevice);

    // Gradient along y
    convolution_device(dev_h_odata, dev_after_Gy, dev_grad, nx, ny, 3);

    merging_gradients_device(dev_after_Gx, dev_after_Gy, dev_G, nx, ny);
 
    // Non-maximum suppression, straightforward implementation.
    non_maximum_supression_device(dev_after_Gx, dev_after_Gy, dev_G, dev_nms, nx, ny);

    cudaMemcpy(nms, dev_nms, memSize, cudaMemcpyDeviceToHost);

    // edges with nms >= tmax
    memset(h_odata, 0, sizeof(pixel_t) * nx * ny);
    first_edges(nms, h_odata, nx, ny, tmax);

    // edges with nms >= tmin && neighbor is edge
    bool changed;
    do {
        changed = false;
        hysteresis_edges(nms, h_odata, nx, ny, tmin, &changed);
    } while (changed==true);

    cudaFree(dev_h_odata);
    cudaFree(dev_G);
    cudaFree(dev_after_Gx);
    cudaFree(dev_after_Gy);
    cudaFree(dev_nms);
    cudaFree(dev_grad);

    free(nms);
}

// print command line format
void usage(char *command) 
{
    printf("Usage: %s [-h] [-d device] [-i inputfile] [-o outputfile] [-r referenceFile] [-w windowsize] [-t threshold]\n",command);
}

// main
int main( int argc, char** argv) 
{

    // default command line options
    int deviceId = 0;
    char *fileIn=(char *)"lena.pgm",*fileOut=(char *)"lenaOut.pgm",*referenceOut=(char *)"reference.pgm";
    int tmin = 45, tmax = 50;
    float sigma=1.0f; 

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:i:o:r:n:x:s:h")) !=-1)
    {
        switch(opt)
        {

            case 'd':  // device
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'i': // input image filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }

                fileIn = strdup(optarg);
                break;
            case 'o': // output image (from device) filename 
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                fileOut = strdup(optarg);
                break;
            case 'r': // output image (from host) filename
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                referenceOut = strdup(optarg);
                break;
            case 'n': // tmin
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&tmin)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 'x': // tmax
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&tmax)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 's': // sigma
                if(strlen(optarg)==0 || sscanf(optarg,"%f",&sigma)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
            case 'h': // help
                usage(argv[0]);
                exit(0);
                break;

        }
    }

    // select cuda device
    cutilSafeCall( cudaSetDevice( deviceId ) );
    
    // create events to measure host canny detector time and device canny detector time
    cudaEvent_t startH, stopH, startD, stopD;
    cudaEventCreate(&startH);
    cudaEventCreate(&stopH);
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);

    // allocate host memory
    int* h_idata=NULL;
    unsigned int h,w;

    //load pgm
    if (cutLoadPGMi(fileIn, (unsigned int **)&h_idata, &w, &h) != CUTTrue) {
        printf("Failed to load image file: %s\n", fileIn);
        exit(1);
    }

    // allocate mem for the result on host side
    //int* h_odata = (int*) malloc( h*w*sizeof(unsigned int));
    //int* reference = (int*) malloc( h*w*sizeof(unsigned int));
    
    int* h_odata = (int*) calloc(h*w, sizeof(unsigned int));
    int* reference = (int*) calloc(h*w, sizeof(unsigned int));

    // detect edges at host
    cudaEventRecord( startH, 0 );
    cannyHost(h_idata, w, h, tmin, tmax, sigma, reference);   
    cudaEventRecord( stopH, 0 ); 
    cudaEventSynchronize( stopH );

    // detect edges at GPU
    cudaEventRecord( startD, 0 );
    cannyDevice(h_idata, w, h, tmin, tmax, sigma, h_odata);   
    cudaEventRecord( stopD, 0 ); 
    cudaEventSynchronize( stopD );
    
    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    float timeH, timeD;
    cudaEventElapsedTime( &timeH, startH, stopH );
    printf( "Host processing time: %f (ms)\n", timeH);
    cudaEventElapsedTime( &timeD, startD, stopD );
    printf( "Device processing time: %f (ms)\n", timeD);

    // save output images
    if (cutSavePGMi(referenceOut, (unsigned int *)reference, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", referenceOut);
        exit(1);
    }
    if (cutSavePGMi(fileOut,(unsigned int *) h_odata, w, h) != CUTTrue) {
        printf("Failed to save image file: %s\n", fileOut);
        exit(1);
    }

    // cleanup memory
    cutFree( h_idata);
    free( h_odata);
    free( reference);

    cutilDeviceReset();
}
