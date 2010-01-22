%.hpp:	%.cu
	nvcc -I$(BOOST_ROOT) -I$(GMTL) --cuda $< -o $(<:%.cu=%.hpp)

CUDAFILES = $(shell ls *.cu)
HPPFILES = $(CUDAFILES:%.cu=%.hpp)

CXXFLAGS+= -I$(BOOST_ROOT) -I$(GMTL)
INCLUDES+= -I$(BOOST_ROOT) -I$(GMTL)

all: $(HPPFILES)
