/***************************************************************************
 *   Stefaan Vermassen
 ***************************************************************************/

#include "mpi.h"
#include <cstdlib>
#include <iostream>

using namespace std;

// Creates an array of random numbers. Each number has a value from 0 - 1
double *create_rand_nums(int num_elements, int proc) 
{
  double *rand_nums = (double *)malloc(sizeof(double) * num_elements);
  int i;
  srand(proc);
  for (i = 0; i < num_elements; i++) {
    rand_nums[i] = (rand() / (double)RAND_MAX);
  }
  return rand_nums;
}

unsigned int log2( unsigned int x )
{
  unsigned int ans = 0 ;
  while( x>>=1 ) ans++;
  return ans ;
}


/**
 * Wrapper function around MPI_Allreduce (leave this unchanged)
 * @param sendbuf Send buffer containing count doubles (input)
 * @param recvbuf Pre-allocated receive buffer (output)
 * @param count Number of elements in the send and receive buffers
 */
void allreduce(double *sendbuf, double *recvbuf, int count)
{
	MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/**
 * Wrapper function around MPI_Allreduce (implement reduce-scatter / allgather algorithm)
 * @param sendbuf Send buffer containing count doubles (input)
 * @param recvbuf Pre-allocated receive buffer (output)
 * @param count Number of elements in the send and receive buffers
 */
void allreduceRSAG(double *sendbuf, double *recvbuf, int count)
{
	// number of processes = power of 2
	// count = multiple of P

	int thisProc, nProc;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisProc);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);
	int steps = log2(nProc);
	int counter = 0;

	//reduce-scatter operation
	for (int step=0; step < steps; step++){
		int offset = (thisProc % (1 << (step+1))) < (1 << step)? 1 << step : -(1 << step);
		int sendTo = thisProc + offset;
		double sendToArray[count/2];
		
		if (offset > 0)
		{
			//send upper half
			int j = 0;
			for (int i=count/2; i<count; i++)
			{
				sendToArray[j++] = sendbuf[i];
			}
		}else
		{
			//send under half
			for (int i=0; i<count/2; i++)
			{
				sendToArray[i] = sendbuf[i];
			}		
		}
		MPI_Send(sendToArray, count/2, MPI_DOUBLE, sendTo, 0, MPI_COMM_WORLD);
		MPI_Recv(recvbuf, count/2, MPI_DOUBLE, sendTo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (offset > 0)
		{
			//receive under half
			for (int i=0; i<count/2; i++)
			{
				sendbuf[i] += recvbuf[i];
			}
		
		} else
		{
			//receive upper half
			int j = 0;
			int k = 0;
			for (int i=count/2; i<count; i++)
			{
				sendbuf[j++] = sendbuf[i] + recvbuf[k++];
			}
		}
		counter++;
		count = count/2;
	}

	//(modified) allgather operation
	//count = 1 at start
	for (int step=steps-1; step >= 0; step--){
 		int operationStep = step;
 		int offset = (thisProc % (1 << (step+1))) < (1 << step)? 1 << step : -(1 << step);
 		int sendTo = thisProc + offset;
 		double sendToArray[count];
 		for (int i=0; i<count; i++)
 		{
 				sendToArray[i] = sendbuf[i];
 		}
 		MPI_Send(sendToArray, count, MPI_DOUBLE, sendTo, 0, MPI_COMM_WORLD);
 		MPI_Recv(recvbuf, count, MPI_DOUBLE, sendTo, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 		if (offset > 0)
 		{
 			//receive upper half
 			int j = 0;
 			for (int i=count; i<count*2; i++)
			{
				sendbuf[i] = recvbuf[j++];
			}
 		} else
 		{
 		//receive under half
 			double temp[count];
 			for(int i=0; i<count; i++)
 			{
 				temp[i] = sendbuf[i];
 				sendbuf[i] = recvbuf[i];
 			}
 			//fill the original values
 			int j=0;
 			for (int i=count; i<count*2; i++)
 			{
 				sendbuf[i] = temp[j++];
 			}
 		}	
 		count = count * 2;
 	}
 	for(int i=0; i<count; i++)
 	{
 		recvbuf[i] = sendbuf[i];
 	}
	
	
}

/**
 * Program entry
 */
int main(int argc, char* argv[])
{
	int thisProc, nProc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisProc);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);
	
	// Test code
	double *rand_nums = NULL;
    rand_nums = create_rand_nums(8, thisProc);
	double *recvbuf = (double *)malloc(sizeof(double) * 8);
	double *recvbuftest = (double *)malloc(sizeof(double) * 8);
	allreduce(rand_nums, recvbuftest, 8);
	
	allreduceRSAG(rand_nums, recvbuf, 8);
	for (int i=0; i<8; i++){
		if (recvbuf[i] != recvbuftest[i]){
			printf("RECV: proces= %d, arr[%d]=%f\n", thisProc, i, recvbuf[i]);
			printf("RECV: proces= %d, arr[%d]=%f\n", thisProc, i, recvbuftest[i]);
		}
	}
	free(rand_nums);
	free(recvbuf);
	free(recvbuftest);
	MPI_Finalize();
	exit(EXIT_SUCCESS);
}
