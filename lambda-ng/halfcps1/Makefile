all: factorial

factorial: factorial.cpp constant.hpp ct_if.hpp env.hpp let.hpp print.hpp tags.hpp var.hpp lambda.hpp
	g++ -I/home/jewillco/boost factorial.cpp -o factorial

%.hpp: %.in
	./codegen.tcl $<
