
COMMONFLAGS+= -I../../../.. -I${BOOST_ROOT} -g # --keep

%:	%.cu
	nvcc $? -o $@ --gpu-architecture sm_13 -DMTL_HAS_CUDA ${COMMONFLAGS}

%:	%.cpp
	g++ $? -o $@ ${COMMONFLAGS} 

clean:
	rm -rf *.ii *.cudafe1.*

