#include <omp.h>
#include <stdio.h>

const int OMP_NUM_THREADS = 6;
const int CHUNK = 10;

int main()
{
    omp_set_num_threads(OMP_NUM_THREADS);

    double begin = omp_get_wtime();

    static long nb_pas = 1000000000;
    double pas = 1.0/(double) nb_pas;
    double x, pi = 0.0;
    double som = 0.0;
    int i = 0;

    #pragma omp parallel for reduction (+ : som) firstprivate (x) schedule (static, CHUNK)
    for (i = 0; i < nb_pas; i++)
    {
        x = (i + 0.5) * pas;
        som += 4.0/(1.0 + x * x);
    }
    pi = pas * som;

    double end = omp_get_wtime();

    printf("PI=%.12f\n",pi);
    printf("Time : %f \n", end - begin);

    return 0;
}