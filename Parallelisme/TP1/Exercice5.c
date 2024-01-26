#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

const int OMP_NUM_THREADS = 1;

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

    int l = width < height ? width : height;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double normalizedX = 1.5 * (x - width / 2)/ l;
            double normalizedY = 1.5 * (y - height / 2)/ l;

            double complex Z = normalizedX + I * normalizedY;
            double complex Zi = 0;

            for (int i = 0; i < N; i++)
            {
                Zi = cpow(Z, 2) + C;
                Z = Zi;
            }
            double val = sqrt(pow(creal(Zi), 2.0) + pow(cimag(Zi), 2.0));

            if (val < 200)
            {
                fprintf(file, "255 ");
            } else
            {
                fprintf(file, "0 ");
            }
            
        }
        fprintf(file, "\n");
    }

    fclose(file);

    return 0;
}