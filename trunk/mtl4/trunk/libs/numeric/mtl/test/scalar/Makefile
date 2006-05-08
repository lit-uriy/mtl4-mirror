CXXFLAGS += -I${BOOST_ROOT} -I${MTL_BOOST_ROOT} # -I${GLAS_ROOT}/branches/state_2006_2_23/glas 

CXX = conceptg++

TARGETS =	age_example1	age_example2

it:	all

all:	${TARGETS}

age_example1:	age_example1.cpp 

age_example2:	age_example2.cpp 

concepts_hpp:	${MTL_BOOST_ROOT}/boost/numeric/mtl/scalar/concepts.hpp

clean:
	rm -rf ${TARGETS}