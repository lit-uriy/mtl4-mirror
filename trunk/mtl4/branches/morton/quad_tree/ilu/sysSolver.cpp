/*****************************************************************************
  file: sysSolver.cpp
  -----------------
  Solving system of linear equations: Ax=b
  	--> incomplete LU +
  		fixed point iteration w/ forward and backward substitutions

  contains Main function

  Revised on: 07/27/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/

#include "sysSolver.h"

//_______________________ MAIN _________________________________

int main(int argc, char** argv)
{
	int numInputMat, 	//number of matrices in input
		nRows, nCols, 	//dimensions
		sym=0;			//is matrix symmetric?
	char fmt[MAX_FLAG_SIZE];	//temp storage for format

	//get user arguments
  	getUserArgs(argc, argv);

	//read in number of input matrices from std in
	// ./mkInput, whose output is piped into ./sysSolver as input,
	// prints this out before anything else.
	// This just makes sure that that data is read.
	scanf("%d",&numInputMat);
	assert (numInputMat == 1);

	// Now, let's read in the data that really matter

	//get nRows, nCols, and fmt, and set sym if needed.
	scanf("%d", &nRows);
	scanf("%d", &nCols);
	scanf("%s", fmt);
	if(strcmp (fmt, HBS) == 0)
		sym = 1;

//printf("A0\n");
	//create the stocks for baseBlocks and quadNodes
	// --> determines how many "data" (quadNodes or baseBlocks)
	//		to allocate at a time
	// --> sets the stepping into the memory space allocated for base blocks
	createStocks();
//printf("A\n");
	//define Matrix - will set the maximum allocation requirement,
	// which is the number of "data" that will ever be needed for the given
	// matrix.
	mat.init(nRows, nCols, sym);

//printf("B\n");
	// initialize the stocks, by calling initMem() for each stock type.
	initStocks();

//printf("C\n");
	//Get Data and build the quadtree
	numb_bblocks = 0;
	buildTree(fmt);
//printf("D\n");
	readRhs();

//printf("E\n");
	//solve system of equations
	solveSys();

//printf("F\n");
	//destroy matrix
	mat.kill();
	
//printf("G\n");
	//destroy stocks
	destroyStocks();

//printf("H\n");
	return 0;
}


//____________________ FUNCTION DEFINITIONS _______________________

/*
void getUserArgs(int nargc, char** nargv)
	get the user arguments from the command line
	the args include printing in/out flags and their formats,
	the baseOrderLg flag and the value, etc... printHelp for more info
*/
void getUserArgs(int nargc, char** nargv)
{
	int arg = 1;

	if( (nargc%2) == 0){
		printHelp();
		exit(1);
	}

	//get flags and values, if any
	while ((nargc-arg) > 0)
	{
		setFlag(nargv[arg], nargv[arg+1]);
		arg += 2;
	}

	//computer base order and size - related variables
	baseOrder = powerOf2(baseOrderLg);
	baseSize = powerOf4(baseOrderLg);
	baseOrder2 = baseOrder * 2;
	baseSize2 = baseSize *2;
}

/*
void setFlag(char* flagName, char *value)
	get a user entered flag, and set the appropriate value.
*/
void setFlag(char* flagName, char *value)
{
	//baseOrderLh
	if(strcmp(flagName, BASE_ORDER_LG)==0){
		baseOrderLg = atoi(value);
	}
	//damping factor
	else if(strcmp(flagName, DAMP_FACT)==0){
		OMEGA = atof(value);
printf("OMEGA = %lf\n", OMEGA);
	}
	//input printing
	else if(strcmp(flagName, PRINT_INPUT)==0){
		printInput = 1;
		checkFormat(value);
		strcpy(printINformat, value);
	}
	//output printing
	else if(strcmp(flagName, PRINT_OUTPUT)==0){
		printOutput = 1;
		checkFormat(value);
	}
	// in/output printing
	else if(strcmp(flagName, PRINT_ALL)==0){
		printInput = 1;
		printOutput = 1;
		checkFormat(value);
		strcpy(printINformat, value);
	}
	//else, error...
	else{
		printf("Invalid Flag:  %s \n",flagName);
		exit(1);
	}
}

/*
void checkFormat(char* fmt)
	Cheching if a user entered format is a valid one.
	Print and error and exit if not.
*/
void checkFormat(char* fmt)
{
	int i;
	for(i=0; i<nFormats; i++){
		if(strcmp(formats[i].name, fmt)==0)
			break;
	}
	if(i==nFormats){
		printf("Invalid Format:  %s\n",fmt);
		printHelp();
		exit(1);
	}
}

/*
void printHelp()
	print flags and formats info for user
*/
void printHelp()
{
	int i;
	printf("\nsyntax: ./matOps {[flag] [value]}* \n");
	printf("----------------------------------------------------\n\n");
	printf("    * --> 0 or more sequences of [flag] and [value] \n\n");

	printf("\t Flags \n");
	printf("\t --------\n");
	for(i=0; i<nFlags; i++){
		printf("\t%s\t%s\n", flags[i].name, flags[i].description);
	}
	printf("\n");
	printf("\t Formats \n");
	printf("\t ----------\n");
	for(i=0; i<nFormats; i++){
		printf("\t%s\t%s\n", formats[i].name, formats[i].description);
	}
	printf("\n");
}

/*
void buildTree(char* fmt)
	determine which tree building function to call depending on input format
*/
void buildTree(char* fmt)
{
	if(strcmp(fmt, ROW_MAJ_Z)==0){
		mat.buildTree_RM();
	}
	else if( (strcmp(fmt, HB)==0) || (strcmp(fmt, HBS)==0) ){
		mat.buildTree_HB();
	}
	else{
		printf("Invalid Input Format:  %s\n",fmt);
		exit(-17);
	}
}

/*
void printMatrix(Both node, char* printfmt)
	determine which matrix printing function to call
		depending on specified format.
*/
void printMatrix(Both node, char* printfmt)
{
	if(strcmp(printfmt,ROW_MAJ_Z)==0){
		mat.printRowMaj(node);
	}
	else if(strcmp(printfmt,TREE_Z)==0){
		mat.printTree(node, 0, LEVEL_START);
	}
}

/*
void solveSys()
	Solves the system of linear equations:
	- backs up a copy of the original matrix before performing the incomplete
		LU decomposition since this is done destructively.
	- Then, calls the iterative solver...
	- Finally, checks the result and print out the number of iterations needed
		to find the solution.
	Timing is performed only during inc LUD and iteration.
*/
void solveSys()
{
	//prepare the array that will hold the final result
	dataType* result = (dataType*)(valloc(mat.getRows() * sizeof(dataType)));
	int iter;

	//save a copy of original matrix
	mat.copyMatrix(&(mat.backupNode), mat.mainNode, LEVEL_START);

	//print input
	if(printInput){
		printf("\nMatrix:\n");
		printMatrix(mat.mainNode, printINformat);
		printf("\n");

		printf("\nRhs:\n");
		printArray(mat.rhs, mat.getRows() );
		printf("\n");
  }
  
/*
  	//LUD
	dataType* lu_res = (dataType*)(valloc(mat.getRows() * sizeof(dataType)));
	dataType* ul_res = (dataType*)(valloc(mat.getRows() * sizeof(dataType)));
	LUD( &(mat.mainNode),
			MTN_START, BND_PART_ALL, BND_PART_ALL , LEVEL_START );
	lu_substitut_test( mat.backupNode, mat.mainNode, lu_res );
	//ULD
	mat.freeNodes(&(mat.mainNode), LEVEL_START);
	mat.copyMatrix(&(mat.mainNode), mat.backupNode, LEVEL_START);
	ULD( &(mat.mainNode),
			 MTN_START, BND_PART_ALL, BND_PART_ALL , LEVEL_START );
	ul_substitut_test(mat.backupNode, mat.mainNode, ul_res);
	//check result. Print if fails.
	for( int i=0; i<mat.getRows(); i++ ){
		if( abs(lu_res[i] - ul_res[i]) > 0.000001 ) {
			printf(" RESULT CHECK FAILED!!! at index %d \n\n", i);
			printf("\t%lf\t%lf\n", lu_res[i], ul_res[i]);
			break;
		}
	}
	
  printf(" RESULT CHECK PASSED!!! \n\n");
  
  //for(int i=0; i<mat.getRows(); i++)
   // printf("\t%lf\t%lf\n", lu_res[i], ul_res[i]);
    
  free(lu_res);
  free(ul_res);
*/

	timestamp_start("");
		//LU Decomposition with pivoting
mat.numb_fill_in = 0;
printf(" Number of baseblocks - start:  %d\n", numb_bblocks);
		LUD( &(mat.mainNode),
				MTN_START, BND_PART_ALL, BND_PART_ALL , LEVEL_START );
printf(" Number of fill ins generated:  %d\n", mat.numb_fill_in);	
printf(" Number of baseblocks - end:  %d\n", numb_bblocks);
	/*			
		printf("\nMatrix:\n");
		printMatrix(mat.mainNode, printINformat);
		printf("\n");
	*/	
		
	//timestamp_end("LU Decomposing - TIME TAKEN:");
		//Solve Ax = b
		lu_sysSolver(mat.backupNode, mat.mainNode, result, (int*)(&iter));
	timestamp_end("SOLVING SYS OF EQs. - TIME TAKEN:");

	int nans = 0;
	for(int i=0; i<mat.getRows(); i++){
		if(isnan(result[i]))
			nans++;
	}

	//print result and number of iterations
	printf(
		"\n\tResult returned after %d iterations - nans: %d\n\n", iter, nans);
	if(printOutput){
		printArray(result, mat.getRows() );
		printf("\n\n");
	}
	free(result);
}


//________________________________ END ____________________________________________________
///////////////////////////////////////////////////////////////////////////////////////////

