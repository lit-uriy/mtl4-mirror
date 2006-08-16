
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace std;

int main(int argc, char** argv)
{
	ifstream input;
	input.open ( argv[1] );

	if ( !input )
	{
		printf("Error...Opening file %s FAILED!!!\n", argv[1] );
		return 0;
	}

	char line[100];

	//read first 4 lines
	input.getline( line, sizeof(line) );
	cout<<line<<endl;
	input.getline( line, sizeof(line) );
	cout<<line<<endl;
	input.getline( line, sizeof(line) );
	cout<<line<<endl;
	input.getline( line, sizeof(line) );
	cout<<line<<endl;

	while(!input.eof()) {
		input.getline( line, sizeof(line) );
		cout<<line+1<<endl;
	}

	input.clear ( );
	input.close ( );

	return 0;
}

////////////////////////////////////////////////////////////////////////

