all: factorial

factorial: factorial.cpp constant.hpp ct_if.hpp env.hpp let.hpp print.hpp tags.hpp var.hpp lambda.hpp begin.hpp set.hpp ifwhile.hpp
	g++ -O3 -S -I/home/jewillco/boost factorial.cpp
	g++ -O3 -I/home/jewillco/boost factorial.cpp -o factorial

%.hpp: %.in codegen.tcl
	./codegen.tcl $<
