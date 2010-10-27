project(MTL-tests)

cmake_minimum_required(VERSION 2.6)
#let cmake look for the svn-version
find_package(Subversion)
if(Subversion_FOUND)
	Subversion_WC_INFO(${CMAKE_CURRENT_BINARY_DIR} mtlSubinfo)
	set(CurrentRevision ${mtlSubinfo_WC_REVISION})
#	message("current revision: ${mtlSubinfo_WC_REVISION}")
else(Subversion_FOUND)
	set(CurrentRevision "0")
endif(Subversion_FOUND)

SET(USER_PROJECT_NAME "mtl_test")
SET(USER_CUR_DIR "/usr/share/mtl/test")

configure_file(CMakeLists_tests.in CMakeLists_test.txt @ONLY)
configure_file(preinst.in preinst @ONLY)

INSTALL(FILES CMakeLists_test.txt
	DESTINATION test
	RENAME CMakeLists.txt
       )

INSTALL(DIRECTORY ../../libs/numeric/mtl/test
	DESTINATION .
	FILES_MATCHING PATTERN "*.cpp"
	PATTERN "*.h"
	PATTERN "*.c"
	PATTERN "*.mtx"
	PATTERN "README"
       )	

SET(CPACK_PACKAGE_NAME "MTL-tests")
SET(CPACK_PACKAGE_VENDOR "Peter Gottschling & Andrew Lumsdaine")
SET(CPACK_PACKAGE_VERSION "4.0.${CurrentRevision}")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "This package contains the MTL tests.")
SET(CPACK_PACKAGE_CONTACT "mtl@osl.iu.edu")
SET(CPACK_RESOURCE_FILE_LICENSE "../../license.mtl.txt")
SET(CPACK_RESOURCE_FILE_README "../../README")
SET(CPACK_PACKAGING_INSTALL_PREFIX "/usr/share/mtl")

SET(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_CURRENT_BINARY_DIR}/postrm")
SET(CPACK_RPM_PACKAGE_ARCHITECTURE "noarch")
SET(CPACK_RPM_PACKAGE_DESCRIPTION ${CPACK_PACKAGE_DESCRIPTION_SUMMARY})
SET(CPACK_RPM_PACKAGE_REQUIRES "mtl")

#the debian way..
SET(CPACK_GENERATOR "DEB" "RPM" "TGZ" "TBZ2")
SET(CPACK_BINARY_DEB TRUE)
SET(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "all")
SET(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA "${CMAKE_CURRENT_SOURCE_DIR}/preinst" "${CMAKE_CURRENT_SOURCE_DIR}/postrm")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "mtl")
#SET(CPACK_DEBIAN_PACKAGE_CONFLICTS "pmtl")
INCLUDE(CPack)
