all: factorial

COMPILER = icc
OPTS = -O3 -fno-alias
BOOST = /u/jewillco/boost

factorial: factorial.cpp constant.hpp ct_if.hpp env.hpp let.hpp print.hpp tags.hpp var.hpp lambda.hpp begin.hpp set.hpp ifwhile.hpp Makefile
	$(COMPILER) $(OPTS) -S -I$(BOOST) factorial.cpp
	$(COMPILER) $(OPTS) -I$(BOOST) factorial.cpp -o factorial

%.hpp: %.in codegen.tcl
	./codegen.tcl $<
