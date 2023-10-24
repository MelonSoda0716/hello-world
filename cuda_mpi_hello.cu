#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdint.h>

/*
$ nvcc cuda_mpi_hello.cu -o cuda_mpi_hello \
-I$MPI_ROOT/include \
-I$CUDA_HOME/include \
-L$MPI_ROOT/lib -lmpi
$ mpiexec -x UCX_LOG_LEVEL=INFO ./cuda_mpi_hello
*/

__global__ void initialize_one(int *array, int N){

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;

   for(int i = idx; i < N; i = i + stride){
      array[i] = 1;
   }

}

__global__ void initialize_zero(int *array, int N){

   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;

   for(int i = idx; i < N; i = i + stride){
      array[i] = 0;
   }

}

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

   cudaStream_t stream;
   cudaDeviceProp device_prop;

   cudaSetDevice(local_rank);
   cudaGetDeviceProperties(&device_prop, local_rank);

   int N = 1000;
   int *host_array, *gpu_array;
   cudaMallocHost((void**)&host_array, sizeof(int) * N);
   cudaMalloc((void**)&gpu_array, sizeof(int) * N);

   for(int i = 0; i < N; i++){
      host_array[i] = 0;
   }

   cudaStreamCreate(&stream);
   if(rank == 0){
      initialize_one<<< 1,N >>>(gpu_array, N);
   }
   else{
      initialize_zero<<< 1,N >>>(gpu_array, N);
   }
   cudaStreamSynchronize(stream);

   MPI_Bcast(gpu_array, N, MPI_INT, 0, MPI_COMM_WORLD);

   cudaMemcpy(host_array, gpu_array, sizeof(int) * N, cudaMemcpyDeviceToHost);

   printf("%2s: rank=%2d size=%2d device=%d [0x%02x] %s OK=%d\n", hostname, rank, size, local_rank, device_prop.pciBusID, device_prop.name, host_array[N-1]);

   MPI_Finalize();

   cudaFree(host_array);
   cudaFree(gpu_array);

   return 0;

}
