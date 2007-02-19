/*//////////////////////////////////////////////////////////////////
 file mkInput_gen.c

 Sparse matrices representation - Operations


 Larisse D. Voufo
 12/15/2005

/////////////////////////////////////////////////////////////////*/

#include "mkInput.h"


int main(int argc, char** argv)
{
	int rows, cols, i, nRhs=0;

	if(argc != 3){
		printf("\nsyntax: ./mkInput [file_name] [format] \n");
		printf("  Type \"./matOps %s \" for help.\n\n", HELP_OP);
		return -1;
	}

	//print number of matrices
	printf("1  \n");
	//get Headers
	if(strcmp(argv[2],ROW_MAJ_Z)==0){
		if(!(getRowsAndCols_RM(argv[1], &rows, &cols)))
			return -1;
	}
	else if( (strcmp(argv[2],HB)==0) ||
				(strcmp(argv[2],HBS)==0) ){
		if(!(getMatInfo_HB(argv[1], &rows, &cols, &nRhs)))
			return -1;
	}
	else{
		printf("Invalid Input Format:  %s\n",argv[2]);
		return -1;
	}

	//print header
	printf("%d  %d  %s  \n", rows, cols, argv[2]);

	//get values
	if(strcmp(argv[2],ROW_MAJ_Z)==0){
		if(!(getInput_RM(argv[1])))
			return -1;
	}
	else if( (strcmp(argv[2],HB)==0) ||
				(strcmp(argv[2],HBS)==0) ){
		if(!(getInput_HB(argv[1])))
			return -1;
	}

	//get RHS
	if(nRhs) {
		double* rhs;
		readHB_newaux_double(argv[1], 'F', &rhs);
		for(i=0; i<rows; i++) {
			printf("\t%G", rhs[i]);
			if(((i+1)%10) == 0)
				printf("\n");
		}
	}
	else {
		srand(time(NULL));
		//printf("\n");
		for(i=0; i<rows; i++){
			//printf("\t%lf", ( ((double)(rand()%32766))/200 ) / FACT );
			printf("\t0.0555550");
			if(((i+1)%10) == 0)
				printf("\n");
		}
	}
	printf("\n");

	return 0;
}


int getMatInfo_HB(char* filename, int *rows, int *cols, int *nRhs)
{
	char* type;
	int nnzero;

	readHB_info(filename, rows, cols, &nnzero, &type, nRhs);

	//avoid pattern and complex inputs, for now
	if( (type[0]=='P') || (type[0]=='C' ) ) {
		printf("getMatInfo_HB: Error... Invalid Input Matrix Type\n");
		return 0;
	}

	return 1;
}


int getInput_HB ( char *filename )
{
	int nrow, ncol, nnzero;
	int* colptr=NULL;
	int* rowind=NULL;
	double* values=NULL;

	readHB_newmat_double(filename, &nrow, &ncol, &nnzero,
							&colptr, &rowind, &values);

  	int t; double val;
  	//print nnzero, colptr, rowind, and values
  	printf("%d \n", nnzero);
  	for(t=0; t<ncol+1; t++) {
  		printf("%d  ", colptr[t]);
  		if(((t+1)%16)==0)
  			printf("\n");
	}
  	printf("\n");
  	for(t=0; t<nnzero; t++) {
  		printf("%d  ", rowind[t]);
  		if(((t+1)%16)==0)
  			printf("\n");
	}
  	printf("\n");
  	for(t=0; t<nnzero; t++) {
  		val = values[t]/FACT;
  		if (val == 0.0)
  			val = 1.0;
  		printf("%G  ", val/FACT);
  		if(((t+1)%10)==0)
  			printf("\n");
	}
  	printf("\n");

	if ( colptr )
		delete [] colptr;
	if ( rowind )
		delete [] rowind;
	if ( values )
		delete [] values;
	return 1;
}

int getRowsAndCols_RM(char* fileName, int *rows, int *cols)
{
	ifstream input;
	input.open ( fileName );

	if ( !input )
	{
		printf("Error...Opening file %s FAILED!!!\n", fileName);
		return 0;
	}

	input>>(*rows);
	input>>(*cols);

	input.clear ( );
	input.close ( );

	return 1;
}


int getInput_RM ( char *fileName)
{
	ifstream input;
	double val;
	int i,j, rows, cols;

	input.open ( fileName );
	if ( !input )
	{
		printf("Error...Opening file %s FAILED!!!\n", fileName);
		return 0;
	}

	input>>rows;
	input>>cols;

	for(i=0; i<rows; i++){
		for(j=0; j<cols; j++){
			input>>val;
			printf("%G ", val/FACT);
		}
		printf("\n");
	}
	input.clear ( );
  input.close ( );

	return 1;
}





//________________________________ END ____________________________________________________
///////////////////////////////////////////////////////////////////////////////////////////
