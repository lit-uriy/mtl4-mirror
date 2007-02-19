/*//////////////////////////////////////////////////////////////////
 file mkInput.c

 Sparse matrices representation - Operations

 Larisse D. Voufo
 12/15/2005

/////////////////////////////////////////////////////////////////*/

#include <stdlib.h>      /* defines NULL = 0L  */
#include <stdio.h>
#include <unistd.h>
#include <malloc.h>
#include <math.h>

using namespace std;

int main(int argc, char** argv)
{

	if(argc != 2){
		printf("\nsyntax: ./mk_spdMat [order] \n");
		return -1;
	}
	int order = atoi(argv[1]);

	printf("%d  %d\n", order, order );

	double *temp;
	int alloc_size,i,j;
	alloc_size = order*order;
	temp = (double*)valloc(alloc_size*sizeof(double));
	if(temp==NULL) {
		printf("Error: could not allocate space for order %d matrix",order);
		exit(1);
	}
	/*srand(time(NULL));*/
	srand(1);
	int first = (int)sqrt(order), last = first + 2;
	for(i=0;i<order;i++) {
		for(j=0;j<=i;j++) {
			if(i==j) {
				temp[i*order+j] = 6.1;
			} else {
			  if (abs(i - j) >= first && abs(i - j) <= last) 
					temp[i*order+j] = temp[j*order+i] = -1.0;
				else
					temp[i*order+j] = temp[j*order+i] = 0.0;
			}
		}
	}
	/*
	srand(1);
	int first = (int)sqrt(order), last = first + 2;
	for(i=0;i<order;i++) {
		for(j=0;j<=i;j++) {
			if(i==j) {
				temp[i*order+j] = (double)order+1;
			} else {
			  if (abs(i - j) >= first && abs(i - j) <= last) 
				// if( (i < 5)||(j<5) )
					temp[i*order+j] = temp[j*order+i] = ((double)(5555%100000))/100000.0;
				else
					temp[i*order+j] = temp[j*order+i] = 0.0;
			}
		}
	}*/

	for(i=0; i<alloc_size; i++){
		printf("%lf ", temp[i]);
		if( ((i+1)%order) == 0 )
			printf("\n");
	}

	return 0;
}



//________________________________ END ____________________________________________________
///////////////////////////////////////////////////////////////////////////////////////////
