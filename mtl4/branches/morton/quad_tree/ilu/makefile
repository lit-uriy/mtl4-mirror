


CC =        gcc
CXX =       g++

# flags for g++
# problems with -static on MacOS ...
FLAGS=   -O3 -finline-limit=2000 -Winline -I/usr/include/malloc -dynamic

# flags for icc
#FLAGS="-O3 -ipo -no-prec-div -static -ip -xN"

CXXFLAGS = $(FLAGS)
CFLAGS = $(FLAGS)

EXE =   mk_spdMat mkInput sysSolver 

it:	$(EXE)

clean:
	rm -f *~ *.o $(EXE)

mkInput: mkInput.o iohb.o
	$(CXX) $(FLAGS) $^ -o $@

sysSolver: iohb.o bounds.o matrix.o iLU.o iUL.o sysSolve.o  sysSolver.o 
	$(CXX) $(FLAGS) $^ -o $@