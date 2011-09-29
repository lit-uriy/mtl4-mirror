############################
#This configuration file defines some cmake variables:
#MTL_INCLUDE_DIRS: list of include directories for the mtl
#MTL_LIBRARIES: libraries needed for interfaces like umfpack and arprec, see below
#MTL_COMPILE_DEFINITIONS: definitions to enable the requested interfaces
#
#supported components:
#Umfpack, Arprec

SET(MTL_INCLUDE_DIRS "${MTL_DIR}/../../include")
find_package(Boost 1.36 REQUIRED)
if(Boost_FOUND)
	LIST(APPEND MTL_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
endif(Boost_FOUND)
#message("find components: ${MTL_FIND_COMPONENTS}")
#we found nothing..
set(MTL_NOT_FOUND )
foreach(CURCOMP ${MTL_FIND_COMPONENTS})
#look for a file called cmake/COMPONENT.cmake in the mtl-directory (/usr/share/mtl/)
	string(TOUPPER ${CURCOMP} CURCOMP_UPPER)
	set(curfile "${MTL_DIR}/cmake/${CURCOMP_UPPER}.cmake")
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
