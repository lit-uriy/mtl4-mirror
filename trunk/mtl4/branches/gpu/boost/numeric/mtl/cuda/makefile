%.cuhpp:	%.cu
	nvcc -I$(BOOST_ROOT) -I$(GMTL) --cuda $< -o $(<:%.cu=%.cuhpp)

%.hpp:	%.cuhpp
	 cuhpp2hpp $< 

CUDAFILES = $(shell ls *.cu)
CUHPPFILES = $(CUDAFILES:%.cu=%.cuhpp)
HPPFILES = $(CUDAFILES:%.cu=%.hpp)


all: $(HPPFILES)

clean:
	rm -f $(HPPFILES) $(CUHPPFILES)