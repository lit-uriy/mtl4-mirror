all: lettest

lettest: lettest.cpp constvar.hpp let.hpp env.hpp nodes.hpp
	g++ -o lettest lettest.cpp

constvar.hpp: constvar.in codegen.tcl
	./constvar.in > constvar.hpp

let.hpp: let.in codegen.tcl
	./let.in > let.hpp

clean:
	rm -f constvar.hpp let.hpp
