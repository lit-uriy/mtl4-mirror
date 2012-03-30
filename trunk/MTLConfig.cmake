############################
#This configuration file defines some cmake variables:
#MTL_INCLUDE_DIRS: list of include directories for the mtl
#MTL_LIBRARIES: libraries needed for interfaces like umfpack and arprec, see below
#MTL_COMPILE_DEFINITIONS: definitions to enable the requested interfaces
#
#supported components:
#Umfpack, Arprec

find_package(Boost 1.42 REQUIRED)
if(Boost_FOUND)
	LIST(APPEND MTL_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)

include(${MTL_DIR}/tools/cmake/Vampir.cmake)
include(${MTL_DIR}/tools/cmake/UMFPACK.cmake)
include(${MTL_DIR}/tools/cmake/ARPREC.cmake)
unset(MTL_LIBRARIES )

if(HAVE_UMFPACK)
	add_definitions("-DMTL_HAS_UMFPACK")
	include_directories(${UMFPACK_INCLUDE_DIRS})
	list(APPEND MTL_LIBRARIES ${UMFPACK_LIBRARIES})
endif()
if(HAVE_ARPREC)
	add_definitions("-DMTL_HAS_ARPREC")
	include_directories(${ARPREC_INCLUDE_DIRS})
	list(APPEND MTL_LIBRARIES ${ARPREC_LIBRARIES})
endif()
if(EXISTS ${MTL_DIR}/boost/numeric/mtl/mtl.hpp)
	SET(MTL_INCLUDE_DIRS "${MTL_DIR}")
else()
	SET(MTL_INCLUDE_DIRS "${MTL_DIR}/../../include")
endif(EXISTS ${MTL_DIR}/boost/numeric/mtl/mtl.hpp)

if(ENABLE_VAMPIR AND VAMPIR_FOUND)
#add_definitions("-DMTL_HAS_VPT -DVTRACE -vt:inst manual")
	add_definitions("-DMTL_HAS_VPT -DVTRACE")
	add_definitions(${VT_COMPILE_FLAGS})
	set(MTL_LINK_FLAGS "${MTL_LINK_FLAGS} ${VT_LINK_FLAGS}")
	if(EXISTS ${MTL_DIR}/boost/numeric/mtl/interface/vpt.cpp)
		add_library(mtl_vampir ${MTL_DIR}/boost/numeric/mtl/interface/vpt.cpp)
	else()
		add_library(mtl_vampir ${MTL_DIR}/vpt.cpp)
	endif()
	list(APPEND MTL_LIBRARIES ${VT_LIBRARIES} mtl_vampir)
	set(HAVE_VAMPIR TRUE)
endif(ENABLE_VAMPIR AND VAMPIR_FOUND)

#message("find components: ${MTL_FIND_COMPONENTS}")
#we found nothing..
set(MTL_NOT_FOUND )
#remove?
foreach(CURCOMP ${MTL_FIND_COMPONENTS})
#look for a file called COMPONENT.cmake in the mtl-directory (/usr/share/mtl/)
	string(TOUPPER ${CURCOMP} CURCOMP_UPPER)
	set(curfile "${MTL_DIR}/tools/cmake/${CURCOMP_UPPER}.cmake")
	if(EXISTS ${curfile})
		include(${curfile})
		#look for component 
		#check if the component was correctly found
		if(HAVE_${CURCOMP_UPPER})
			list(APPEND MTL_INCLUDE_DIRS ${${CURCOMP_UPPER}_INCLUDE_DIRS})
			list(APPEND MTL_LIBRARIES ${${CURCOMP_UPPER}_LIBRARIES})
			list(APPEND MTL_COMPILE_DEFINITIONS "-DMTL_HAS_${CURCOMP_UPPER}")
		else()
			list(APPEND MTL_NOT_FOUND ${CURCOMP})
		endif()
	else()
		list(APPEND MTL_NOT_FOUND ${CURCOMP})
	endif()
endforeach()

if(MTL_FIND_REQUIRED AND MTL_NOT_FOUND)
	message(SEND_ERROR "could not find all components: ${MTL_NOT_FOUND}")
endif()
include_directories(${MTL_INCLUDE_DIRS})
