#include <omp.h>
#include <stdio.h>

const int OMP_NUM_THREADS = 4;

int main()
{
    omp_set_num_threads(OMP_NUM_THREADS);

    #pragma omp parallel for
    for (int i = 1; i < 51; i++)
    {
        int idThread = omp_get_thread_num();
        printf("From Thread %d : i = %d \n", idThread, i);
    }

    return 0;
}