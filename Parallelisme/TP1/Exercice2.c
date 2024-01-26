#include <omp.h>
#include <stdio.h>

const int OMP_NUM_THREADS = 4;

int main()
{
    omp_set_num_threads(OMP_NUM_THREADS);
    int val1 = 1000;
    int val2 = 2000;

    #pragma omp parallel firstprivate (val2)
    {
        int idThread = omp_get_thread_num();
        printf("Thread %d is here \n", idThread);
        val2++;
        printf("From Thread %d : val2 = %d \n", idThread, val2);
    }

    return 0;
}