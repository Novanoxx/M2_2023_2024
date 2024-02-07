#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

const int OMP_NUM_THREADS = 4;  // Adjust the number of threads as needed

int main()
{
    omp_set_num_threads(OMP_NUM_THREADS);

    double complex C = -0.8 + I * 0.156;

    int N = 200;
    int width = 600;
    int height = 400;

    FILE *file;
    file = fopen("out.pgm", "wb");
    fprintf(file, "P2\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");

    double begin = omp_get_wtime();
    int l = width < height ? width : height;

    int *pixelValues = (int *)malloc(width * height * sizeof(int));

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double normalizedX = 1.5 * (x - width / 2) / l;
            double normalizedY = 1.5 * (y - height / 2) / l;

            double complex Z = normalizedX + I * normalizedY;
            double complex Zi = 0;

            for (int i = 0; i < N; i++)
            {
                Zi = cpow(Z, 2) + C;
                Z = Zi;
            }
            double val = sqrt(pow(creal(Zi), 2.0) + pow(cimag(Zi), 2.0));

            pixelValues[y * width + x] = (val < 200) ? 255 : 0;
        }
    }

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            fprintf(file, "%d ", pixelValues[y * width + x]);
        }
        fprintf(file, "\n");
    }
    double end = omp_get_wtime();
    printf("Time : %f \n", end - begin);

    fclose(file);
    free(pixelValues);

    return 0;
}
