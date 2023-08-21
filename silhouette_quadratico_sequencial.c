#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

#define DATASET "iris.data"
#define NUM_CLUSTERS 3
#define NUM_FEATURES 4
#define MAX_INSTANCES 150

typedef struct {
    double features[NUM_FEATURES];
    int cluster;
} Instance;

__device__ double euclidean_distance(double *a, double *b, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

__global__ void silhouette_kernel(Instance *dataset, double *results, int num_instances, int num_features, int num_clusters) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < num_instances) {
        double a = 0.0;
        double b = DBL_MAX;
        int cluster_count[NUM_CLUSTERS] = {0};
        double avg_distances[NUM_CLUSTERS] = {0};

        for (int j = 0; j < num_instances; j++) {
            if (i == j) continue;

            double distance = euclidean_distance(dataset[i].features, dataset[j].features, num_features);
            avg_distances[dataset[j].cluster] += distance;
            cluster_count[dataset[j].cluster]++;

            if (dataset[i].cluster == dataset[j].cluster) {
                a += distance;
            }
        }

        if (cluster_count[dataset[i].cluster] > 1) {
            a /= (cluster_count[dataset[i].cluster] - 1);
        }

        for (int c = 0; c < num_clusters; c++) {
            if (c != dataset[i].cluster && cluster_count[c] > 0) {
                double avg_distance = avg_distances[c] / cluster_count[c];
                if (avg_distance < b) {
                    b = avg_distance;
                }
            }
        }

        double silhouette_value = (b - a) / fmax(a, b);
        results[i] = silhouette_value;
    }
}

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

int main() {
    Instance dataset[NUM_INSTANCIAS];
    //read_iris_data(dataset, "iris.csv");
	clock_t start, end;
    double cpu_time_used;

    int num_instances = read_dataset(DATASET, dataset);
    if (num_instances < 0) {
        printf("Erro ao ler o dataset\n");
        return -1;
    }

    Instance *d_dataset;
    double *d_results;
    double *results = (double *)malloc(NUM_INSTANCIAS * sizeof(double));
    cudaMalloc(&d_dataset, NUM_INSTANCIAS * sizeof(Instance));
    cudaMalloc(&d_results, NUM_INSTANCIAS * sizeof(double));

    cudaMemcpy(d_dataset, dataset, NUM_INSTANCIAS * sizeof(Instance), cudaMemcpyHostToDevice);
	start = clock();
    int blockSize = 256;
    int gridSize = (NUM_INSTANCIAS + blockSize - 1) / blockSize;
    silhouette_kernel<<<gridSize, blockSize>>>(d_dataset, d_results, NUM_INSTANCIAS, NUM_FEATURES, NUM_CLUSTERS);
    cudaDeviceSynchronize();

    cudaMemcpy(results, d_results, NUM_INSTANCIAS * sizeof(double), cudaMemcpyDeviceToHost);

    double silhouette_sum = 0.0;
    for (int i = 0; i < NUM_INSTANCIAS; i++) {
        silhouette_sum += results[i];
    }

    double silhouette_avg = silhouette_sum / NUM_INSTANCIAS;
	
	end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	
    printf("Silhouette average: %f\n", silhouette_avg);
	printf("Tempo de execução: %f segundos\n", cpu_time_used);
	
    cudaFree(d_dataset);
    cudaFree(d_results);
    free(results);

    return 0;
}

