all: factorial

OPTS = -O3 -fno-alias

factorial: factorial.cpp constant.hpp ct_if.hpp env.hpp let.hpp print.hpp tags.hpp var.hpp lambda.hpp begin.hpp set.hpp ifwhile.hpp Makefile
	icc $(OPTS) -S -I/u/jewillco/boost factorial.cpp
	icc $(OPTS) -I/u/jewillco/boost factorial.cpp -o factorial

%.hpp: %.in codegen.tcl
	./codegen.tcl $<
