#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DATASET "iris.data"
#define NUM_CLUSTERS 3
#define NUM_FEATURES 4
#define MAX_INSTANCES 150

typedef struct {
    double features[NUM_FEATURES];
    int cluster;
} Instance;

typedef struct {
    double features[NUM_FEATURES];
} Centroid;

void calculate_centroids(Instance *instances, int num_instances, Centroid *centroids) {
    memset(centroids, 0, NUM_CLUSTERS * sizeof(Centroid));
    int cluster_sizes[NUM_CLUSTERS] = {0};

    for (int i = 0; i < num_instances; i++) {
        int cluster = instances[i].cluster;
        for (int j = 0; j < NUM_FEATURES; j++) {
            centroids[cluster].features[j] += instances[i].features[j];
        }
        cluster_sizes[cluster]++;
    }

    
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            centroids[i].features[j] /= cluster_sizes[i];
        }
    }
}

void calculate_general_centroid_from_centroids(Centroid *centroids, Centroid *general_centroid) {
    
    memset(general_centroid, 0, sizeof(Centroid));

    
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            general_centroid->features[j] += centroids[i].features[j];
        }
    }

    
    for (int j = 0; j < NUM_FEATURES; j++) {
        general_centroid->features[j] /= NUM_CLUSTERS;
    }
}

double calculate_interMean(Centroid *centroids, Centroid *general_centroid) {
    double interMean = 0.0;

    for (int i = 0; i < NUM_CLUSTERS; i++) {
        double dist = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            double diff = centroids[i].features[j] - general_centroid->features[j];
            dist += diff * diff;
        }
        interMean += sqrt(dist);
    }

    interMean /= NUM_CLUSTERS;

    return interMean;
}

double calculate_intraMean(Instance *instances, int num_instances, Centroid *centroids) {
    double intraMean = 0.0;

    for (int i = 0; i < num_instances; i++) {
        Centroid cluster_centroid = centroids[instances[i].cluster];
        double dist = 0.0;
        for (int j = 0; j < NUM_FEATURES; j++) {
            double diff = instances[i].features[j] - cluster_centroid.features[j];
            dist += diff * diff;
        }
        intraMean += sqrt(dist);
    }

    intraMean /= num_instances;

    return intraMean;
}

double calculate_BDSilhouette(double interMean, double intraMean) {
    double maxMean = interMean > intraMean ? interMean : intraMean;
    double BDSilhouette = (interMean - intraMean) / maxMean;
    return BDSilhouette;
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
    return i; 
}

void print_instance(Instance instance) {
    printf("Características: ");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%f ", instance.features[i]);
    }
    printf("\nCluster: %d\n", instance.cluster);
}

void print_centroids(Centroid *centroids) {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        printf("Centróide do Cluster %d: ", i);
        for (int j = 0; j < NUM_FEATURES; j++) {
            printf("%f ", centroids[i].features[j]);
        }
        printf("\n");
    }
}

void print_centroid(Centroid centroid) {
    printf("Centróide: ");
    for (int i = 0; i < NUM_FEATURES; i++) {
        printf("%f ", centroid.features[i]);
    }
    printf("\n");
}

int main() {
    clock_t start, end;
    double cpu_time_used;

    Instance* instances = (Instance*) malloc(MAX_INSTANCES * sizeof(Instance));
    Centroid centroids[NUM_CLUSTERS], general_centroid;


    int num_instances = read_dataset(DATASET, instances);
    if (num_instances < 0) {
        printf("Erro ao ler o dataset\n");
        return -1;
    }
    start = clock();
   
    calculate_centroids(instances, num_instances, centroids);

    
    calculate_general_centroid_from_centroids(centroids, &general_centroid);
    end = clock();
    print_centroid(general_centroid);
    print_centroids(centroids);

    double interMean = calculate_interMean(centroids, &general_centroid);
    printf("InterMean: %f\n", interMean);

    double intraMean = calculate_intraMean(instances, num_instances, centroids);
    printf("IntraMean: %f\n", intraMean);

    double BDSilhouette = calculate_BDSilhouette(interMean, intraMean);
    printf("BD-Silhouette: %f\n", BDSilhouette);
    free(instances);


    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Tempo de execução: %f segundos\n", cpu_time_used);

    return 0;
}
