CXXFLAGS+=  -I${MTL_BOOST_ROOT} -I${BOOST_ROOT} -O2 -DNDEBUG

# on Big Red with IBM Compiler
#CXX= xlC

EXECUTABLES= complexity compressed2D_inserter compressed2D dense2D unroll

it: ${EXECUTABLES}

clean:
	rm -f ${EXECUTABLES} *~ *.o