test: test.cpp detail/generated.hpp
	g++3 -O3 -I/u/jewillco/boost_1_28_0 -o test test.cpp

detail/generated.hpp: detail/generator.tcl
	./detail/generator.tcl > detail/generated.hpp
