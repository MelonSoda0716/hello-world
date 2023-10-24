#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

/*
$ mpicc mpi_hello.c -o mpi_hello
$ mpiexec ./mpi_hello
*/

int main(int argc, char *argv[]){
    
    int rank, size, hotname_length;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    MPI_Get_processor_name(hostname, &hotname_length);

    printf("%s: rank=%d size=%d\n", hostname, rank, size);
    
    MPI_Finalize();
    return 0;
    
}
