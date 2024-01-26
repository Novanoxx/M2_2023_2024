#include <omp.h>
#include <stdio.h>

const int OMP_NUM_THREADS = 4;

int main()
{
    omp_set_num_threads(OMP_NUM_THREADS);

    #pragma omp parallel
    {
        int idThread = omp_get_thread_num();
        printf("Thread %d is here \n", idThread);
    }

    return 0;
}