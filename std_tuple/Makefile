test: test.cpp detail/generated.hpp tc2.s
	g++3 -g -I/u/jewillco/boost_1_28_0 -o test test.cpp tc2.s

detail/generated.hpp: detail/generator.tcl
	./detail/generator.tcl > detail/generated.hpp
