#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <stdint.h>

/*
$ nvcc nccl_hello.cu -o nccl_hello \
-I$MPI_ROOT/include \
-I$CUDA_HOME/include \
-I$NCCL_ROOT/include \
-L$MPI_ROOT/lib -lmpi \
-L/$NCCL_ROOT/lib -lnccl
$ mpiexec -x NCCL_DEBUG=INFO ./nccl_hello
*/

static uint64_t getHostHash(const char *string){

   // Based on DJB2, result = result * 33 + char
   uint64_t result = 5381;
   for(int c = 0; string[c] != '\0'; c++){
      result = ((result << 5) + result) + string[c];
   }

   return result;

}

int main(int argc, char *argv[]){

   int rank, size, hotname_length, local_rank=0;
   char hostname[MPI_MAX_PROCESSOR_NAME];

   MPI_Init(&argc, &argv);

   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   MPI_Get_processor_name(hostname, &hotname_length);

   uint64_t hostHashs[size];
   hostHashs[rank] = getHostHash(hostname);
   MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);

   for(int i = 0; i < size; i++){
      if(i == rank){
         break;
      }
      if(hostHashs[i] == hostHashs[rank]){
         local_rank++;
      }
   }

   ncclUniqueId id;
   ncclComm_t comm;
   cudaStream_t stream;
   cudaDeviceProp device_prop;

   if(rank == 0){
      ncclGetUniqueId(&id);
   }
   MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

   cudaSetDevice(local_rank);
   cudaGetDeviceProperties(&device_prop, local_rank);

   cudaStreamCreate(&stream);
   ncclCommInitRank(&comm, size, id, rank);
   cudaStreamSynchronize(stream);

   printf("%2s: rank=%2d size=%2d device=%d [0x%02x] %s\n", hostname, rank, size, local_rank, device_prop.pciBusID, device_prop.name);

   ncclCommDestroy(comm);
   MPI_Finalize();

   return 0;

}
