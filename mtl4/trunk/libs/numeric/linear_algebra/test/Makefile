CXXFLAGS += -I${MTL_BOOST_ROOT} -I${BOOST_ROOT} -g # -L/u/pgottsch/software/conceptgcc/darwin/lib/ppc64/ -Winvalid-pch -m64 --include bits/stdc++.h # -Wno-long-double # 

XTRA_SOURCES = algebraic_functions.hpp ${MTL_BOOST_ROOT}/boost/numeric/linear_algebra/concepts.hpp \
                 Makefile

CXX =  conceptg++

TARGETS =	positive_real	positive_real_explicit positive_real_power string_example \
		float_example int_example mod_n_example accumulation_example \
		concept_map_test      int_example 

it:	all


all:	${TARGETS}

positive_real:	positive_real.cpp ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ positive_real.cpp
 
positive_real_explicit:	positive_real_explicit.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ positive_real_explicit.cpp

positive_real_power:	positive_real_power.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ positive_real_power.cpp

string_example:	string_example.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ string_example.cpp

float_example:	float_example.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ float_example.cpp

int_example:	int_example.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ int_example.cpp

accumulation_example:	accumulation_example.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ accumulation_example.cpp

mod_n_example:	mod_n_example.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ mod_n_example.cpp

concept_map_test:	concept_map_test.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ concept_map_test.cpp

vector_test:	vector_test.cpp  ${XTRA_SOURCES}
	${CXX} ${CXXFLAGS} -o $@ vector_test.cpp

clean:
	rm -rf ${TARGETS} *~