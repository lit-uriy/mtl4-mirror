all: factorial

COMPILER = g++
OPTS = -O3 -fno-alias -finline-limit=10000

factorial: factorial.cpp constant.hpp ct_if.hpp env.hpp let.hpp print.hpp tags.hpp var.hpp lambda.hpp begin.hpp set.hpp ifwhile.hpp Makefile
	$(COMPILER) $(OPTS) -S -I/home/jewillco/boost factorial.cpp
	$(COMPILER) $(OPTS) -I/home/jewillco/boost factorial.cpp -o factorial

%.hpp: %.in codegen.tcl
	./codegen.tcl $<
