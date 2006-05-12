CXXFLAGS += -I${BOOST_ROOT} -I${MTL_BOOST_ROOT} # -I${GLAS_ROOT}/branches/state_2006_2_23/glas 

XTRA_SOURCES = algebraic_functions.hpp ${MTL_BOOST_ROOT}/boost/numeric/linear_algebra/concepts.hpp \
                 Makefile

CXX = conceptg++

TARGETS =	positive_real	positive_real_explicit

it:	all

all:	${TARGETS}

positive_real:	positive_real.cpp ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ positive_real.cpp

positive_real_explicit:	positive_real_explicit.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ positive_real_explicit.cpp

clean:
	rm -rf ${TARGETS}