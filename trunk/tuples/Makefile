BOOST           = /u/jajarvi/boost

#ifeq ($(USER),jajarvi)
#CXX             = ${CXX_COMPILER}
#else
#CXX             = /l/gcc311/bin/g++ -O3
#endif

CXX = /l/gcc320/bin/g++

#CXX = g++3

EXTRAFLAGS      = -pedantic -Wall -Wno-long-long -ftemplate-depth-50

#LIBS            = -lstdc++

#CXX            = KCC
#EXTRAFLAGS     = --strict --display_error_number --diag_suppress 450 --max_pending_instantiations 50 +K0 -g
#LIBS           =

INCLUDES        = -I$(BOOST) 
CXXFLAGS        = $(INCLUDES) $(EXTRAFLAGS)

LIBFLAGS        = $(LIBS)

.SUFFIXES: .cpp .o

SOURCES = \
testi.cpp 

# Create lists of object files from the source file lists.

OBJECTS = ${SOURCES:.cpp=.o}

TARGETS = ${SOURCES:.cpp=.exe}

DEP_FILES = $(SOURCES:.cpp=.dep)



all:    $(TARGETS)

include $(DEP_FILES)

%.exe: %.o
	$(CXX) $(LIBFLAGS) $(CXXFLAGS) -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

%.dep:  %.cpp
	set -e; $(CXX) -M $(INCLUDES) -c $< \
	        | sed 's/\($*\)\.o[ :]*/\1.o $@ : /g' > $@; \
	[ -s $@ ] || rm -f $@


clean:
	rm -rf *.o *.exe *.dep ti_files/*
run:
	./testi.exe

