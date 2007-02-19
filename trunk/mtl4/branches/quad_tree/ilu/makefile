

#CC = gcc
CXX = g++
#CXX = icc

ITL_ROOT = ~/itl/devel/
MTL_STABLE_ROOT = ~/mtl-stable/

FLAGS = -O3 -g -I$(ITL_ROOT) -I$(MTL_STABLE_ROOT)

# flags for g++
# problems with -static on MacOS ...
# FLAGS += -finline-limit=2000 -Winline -I/usr/include/malloc -dynamic

# flags for icc
#FLAGS += -ipo -no-prec-div -static -ip -xN

CXXFLAGS = $(FLAGS)
CFLAGS = $(FLAGS)

EXE =   mk_spdMat mkInput sysSolver itl_test

SRC = iohb.cpp bounds.cpp matrix.cpp iLU.cpp iUL.cpp sysSolve.cpp 
OBJ = $(SRC:.c=.o)

#all: $(OBJ) mk_spdMat mkInput itl_test sysSolver

$(OBJ): $(SRC)
  $(CXX) $(FLAGS) $< -c $@

it:	$(EXE)

clean:
	rm -f $(EXE)

mk_spdMat: mk_spdMat.cpp
	$(CXX) $^ -o $@ $(FLAGS)
	
mkInput: mkInput.cpp iohb.o
	$(CXX) $^ -o $@ $(FLAGS)

itl_test: itl_test.cpp
	$(CXX) $^ -o $@ $(OBJ) $(FLAGS)

sysSolver: sysSolver.cpp
	$(CXX) $^ -o $@ $(OBJ) $(FLAGS)
	
	
clean:
	rm -f *.o *.o~

#----------------------------------------------------------------------------


