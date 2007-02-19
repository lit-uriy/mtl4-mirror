
/* 
  matrix environment setup for incomplete LU
    Global variables are defined in sysSolver.h
*/

#include "env_setup.h"

/* 
  matrix environment setup for incomplete LU
    read in input for 1 matrix...
    set up env (mem stocks for 2 matrices (input + back up)
*/
void env_init(int nargc, char** nargv)
{
	int numInputMat, 	//number of matrices in input
		nRows, nCols, 	//dimensions
		sym=0;			//is matrix symmetric?
	char fmt[MAX_FLAG_SIZE];	//temp storage for format

	//get user arguments
  	getUserArgs(nargc, nargv);

	//read in number of input matrices from std in
	// ./mkInput, whose output is piped into ./sysSolver as input,
	// prints this out before anything else.
	// This just makes sure that that data is read.
  scanf("%d",&numInputMat);
  assert (numInputMat == 1);

  //get nRows, nCols, and fmt, and set sym if needed.
  scanf("%d", &nRows);
  scanf("%d", &nCols);
  scanf("%s", fmt);
  if(strcmp (fmt, HBS) == 0)
    sym = 1;

  //create the stocks for baseBlocks and quadNodes
  // --> determines how many "data" (quadNodes or baseBlocks)
  //to allocate at a time
  // --> sets the stepping into the memory space allocated for base blocks
  createStocks();

  //define Matrix - will set the maximum allocation requirement,
  // which is the number of "data" that will ever be needed for the given
  // matrix.
  mat.init(nRows, nCols, sym);

  // initialize the stocks, by calling initMem() for each stock type.
  initStocks();

  //Get Data and build the quadtree
  numb_bblocks = 0;
  cout<< nRows<<"\t"<<nCols<<endl;
  buildTree(fmt);
}

/*
close environment previously set up by env_init()
 At end of operation, return allocated spaces to available memory.
*/
void env_kill()
{
	//destroy matrix
	mat.kill();
	
	//destroy stocks
	destroyStocks();
}


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
	// baseOrder is defaulted to 3, if not specified
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


#endif

//////////////////////////////////////////////////////////////////////////////
