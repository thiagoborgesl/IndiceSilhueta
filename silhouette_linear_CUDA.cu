#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>


#define DATASET "iris.data"
#define NUM_CLUSTERS 3
#define NUM_FEATURES 4
#define MAX_INSTANCES 150

#define MAX_BLOCKS 1024
#define BLOCK_SIZE 256

#define CUDA_ERROR(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    double sum[NUM_FEATURES];
    int count;
} Centroid;

typedef struct {
    double features[NUM_FEATURES];
    int cluster;
} Instance;

int read_dataset(const char *filename, Instance *instances) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Não foi possível abrir o arquivo %s\n", filename);
        return -1;
    }

    int i = 0;
    while (!feof(file) && i < MAX_INSTANCES) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            if (fscanf(file, "%lf,", &instances[i].features[j]) != 1) {
                printf("Erro ao ler a característica %d da instância %d\n", j+1, i+1);
                fclose(file);
                return -1;
            }
        }
        if (fscanf(file, "%d\n", &instances[i].cluster) != 1) {
            printf("Erro ao ler o cluster da instância %d\n", i+1);
            fclose(file);
            return -1;
        }
        i++;
    }

    fclose(file);
    return i; // Retorna o número de instâncias lidas
}

int* find_cluster_starts(Instance *data, int n_points, int n_clusters) {
    int* cluster_starts = (int*)malloc(n_clusters * sizeof(int));
    if (cluster_starts == NULL) {
        return NULL;
    }

    int current_cluster = data[0].cluster;
    cluster_starts[current_cluster] = 0;

    for (int i = 1; i < n_points; i++) {
        // Se o cluster do ponto atual é diferente do cluster anterior, armazene o índice
        if (data[i].cluster != current_cluster) {
            current_cluster = data[i].cluster;
            cluster_starts[current_cluster] = i+1;
        }
    }

    return cluster_starts;
}

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void calculate_cluster_centroid(int start, int end, Instance *instances, double *centroid, int num_features) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = start + blockIdx.x * blockDim.x + threadIdx.x;

    for (int f = 0; f < num_features; f++) {
        sdata[tid * num_features + f] = (i < end) ? instances[i].features[f] : 0;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int f = 0; f < num_features; f++) {
                sdata[tid * num_features + f] += sdata[(tid + s) * num_features + f];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        for (int f = 0; f < num_features; f++) {
            atomicAddDouble(&centroid[f], sdata[f]);
        }
    }
}

double euclidean_distance(double* a, double* b, int num_features) {
    double sum = 0.0;
    for (int i = 0; i < num_features; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double interMean(double* centroids, double* general_centroid, int num_clusters, int num_features) {
    double sum = 0.0;
    for (int c = 0; c < num_clusters; c++) {
        sum += euclidean_distance(centroids + c * num_features, general_centroid, num_features);
    }
    return sum / num_clusters;
}

__device__ double euclidean_distanceg(double* a, double* b, int num_features) {
    double sum = 0.0;
    for (int i = 0; i < num_features; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__global__ void calculate_intra_cluster_distance(int start, int end, Instance *instances, double *centroid, int num_features, double *result) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = start + blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < end) ? euclidean_distanceg(instances[i].features, centroid, num_features) : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAddDouble(result, sdata[0]);
}


int main() {
    clock_t start, end;
    double cpu_time_used;

    Instance *instances = (Instance*)malloc(sizeof(Instance) * MAX_INSTANCES);

    //leitura do dataset
    int num_instances = read_dataset(DATASET, instances);
    if (num_instances < 0) {
        printf("Erro ao ler o dataset\n");
        return -1;
    }
    /*
    for(int i = 0; i < MAX_INSTANCES; i++) {
        printf("Instance %d, Cluster %d, Features: ", i+1, instances[i].cluster);
        for(int j = 0; j < NUM_FEATURES; j++) {
            printf("%f ", instances[i].features[j]);
        }
        printf("\n");
    }
*/


    double *centroids = (double*)malloc(sizeof(double) * NUM_FEATURES * NUM_CLUSTERS);
    Instance *d_instances;
    double *d_centroids;
    cudaMalloc(&d_instances, sizeof(Instance) * MAX_INSTANCES);
    cudaMemcpy(d_instances, instances, sizeof(Instance) * MAX_INSTANCES, cudaMemcpyHostToDevice);
    cudaMalloc(&d_centroids, sizeof(double) * NUM_FEATURES * NUM_CLUSTERS);
    double *intra_cluster_distances = (double*)malloc(sizeof(double) * NUM_CLUSTERS);
    double *d_intra_cluster_distances;
    cudaMalloc(&d_intra_cluster_distances, sizeof(double) * NUM_CLUSTERS);


    int* cluster_starts = find_cluster_starts(instances, MAX_INSTANCES, NUM_CLUSTERS);
    if (cluster_starts == NULL) {
        printf("Failed to find cluster starts.\n");
        return -1;
    }
    start = clock();
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        int start = cluster_starts[c];
        int end = (c == NUM_CLUSTERS - 1) ? MAX_INSTANCES : cluster_starts[c + 1];
        calculate_cluster_centroid<<<(end - start + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * NUM_FEATURES * sizeof(double)>>>(start, end, d_instances, d_centroids + c * NUM_FEATURES, NUM_FEATURES);

    }
    cudaDeviceSynchronize();
    cudaMemcpy(centroids, d_centroids, sizeof(double) * NUM_FEATURES * NUM_CLUSTERS, cudaMemcpyDeviceToHost);
/*
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        printf("Centroid %d: ", c+1);
        for(int f = 0; f < NUM_FEATURES; f++) {
            printf("%f ", centroids[c * NUM_FEATURES + f]);
        }
        printf("\n");
    }
*/
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        int start = cluster_starts[c];
        int end = (c == NUM_CLUSTERS - 1) ? MAX_INSTANCES : cluster_starts[c + 1];
        int cluster_size = end - start;
        printf("Centroid %d: ", c+1);
        for(int f = 0; f < NUM_FEATURES; f++) {
            centroids[c * NUM_FEATURES + f] /= cluster_size;
            printf("%f ", centroids[c * NUM_FEATURES + f]);
        }
        printf("\n");
    }


    double general_centroid[NUM_FEATURES] = {0.0};

    // Cálculo do centroide geral após a obtenção dos centroides
    for (int f = 0; f < NUM_FEATURES; f++) {
        for (int c = 0; c < NUM_CLUSTERS; c++) {
            general_centroid[f] += centroids[c * NUM_FEATURES + f];
        }
        general_centroid[f] /= NUM_CLUSTERS;
    }

    // Imprimir o centroide geral
    printf("General centroid: ");
    for (int f = 0; f < NUM_FEATURES; f++) {
        printf("%f ", general_centroid[f]);
    }
    printf("\n");

    double inter_mean = interMean(centroids, general_centroid, NUM_CLUSTERS, NUM_FEATURES);
    printf("Inter-cluster mean distance: %f\n", inter_mean);

    cudaMemcpy(d_centroids, centroids, sizeof(double) * NUM_FEATURES * NUM_CLUSTERS, cudaMemcpyHostToDevice);

    for (int c = 0; c < NUM_CLUSTERS; c++) {
      int start = cluster_starts[c];
      int end = (c == NUM_CLUSTERS - 1) ? MAX_INSTANCES : cluster_starts[c + 1];
      calculate_intra_cluster_distance<<<(end - start + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(double)>>>(start, end, d_instances, d_centroids + c * NUM_FEATURES, NUM_FEATURES, d_intra_cluster_distances + c);

    }
    cudaDeviceSynchronize();
    cudaMemcpy(intra_cluster_distances, d_intra_cluster_distances, sizeof(double) * NUM_CLUSTERS, cudaMemcpyDeviceToHost);

    double intra_mean = 0.0;
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        intra_mean += intra_cluster_distances[c];
    }

    intra_mean /= num_instances;

    printf("Intra-cluster mean distance: %f\n", intra_mean);

    double max_distance = (inter_mean > intra_mean) ? inter_mean : intra_mean;
    double bd_silhouette = (inter_mean - intra_mean) / max_distance;
    printf("BD-Silhouette: %f\n", bd_silhouette);

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execução: %lf segundos\n", cpu_time_used);

    cudaFree(d_intra_cluster_distances);
    free(intra_cluster_distances);
    cudaFree(d_instances);
    cudaFree(d_centroids);
    free(centroids);

    return 0;
}