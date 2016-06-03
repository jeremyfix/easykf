
CMAKE_MINIMUM_REQUIRED( VERSION 2.8.4 )

project (easykf) 

SET(PACKAGE_VERSION "2.11")
SET(PACKAGE_LICENSE "GPL")

SET(CMAKE_BUILD_TYPE Release)

SET(PROJECT_DESCRIPTION_SUMMARY "C++ library for Kalman filtering (linear, EKF, UKF)")

#######################################
# Get the libs/inc for the dependencies
#######################################

find_package(PkgConfig)

pkg_check_modules(GSL gsl REQUIRED)
pkg_check_modules(PYTHON python REQUIRED)

set(PKG_CONFIG_DEPENDS "gsl python")

#######################################
# Setting the compilation flags
#######################################

# cflags added by the package
SET(PROJECT_CFLAGS "-std=c++0x")

# cflags added by the pkg-config dependencies contains ';' as separator. This is a fix.
string(REPLACE ";" " " GSL_CFLAGS "${GSL_CFLAGS}")
string(REPLACE ";" " " PYTHON_CFLAGS "${PYTHON_CFLAGS}")

# lib flags added by the package
SET(PROJECT_LIBS "-Wl,--no-as-needed -leasykf")

# libs added by the pkg-config dependencies contains ';' as separator. This is a fix.
string(REPLACE ";" " " GSL_LDFLAGS "${GSL_LDFLAGS}")
string(REPLACE ";" " " PYTHON_LDFLAGS "${PYTHON_LDFLAGS}")

# ldflags required, but not provided by pkg-config
SET(PROJECT_LDFLAGS "")

# Gathering of all flags
# (e.g. for compiling examples)
SET(PROJECT_ALL_CFLAGS  "${PROJECT_CFLAGS} ${GSL_CFLAGS} ${PYTHON_CFLAGS}")
SET(PROJECT_ALL_LDFLAGS "${PROJECT_LIBS} ${PROJECT_LDFLAGS} -L${CMAKE_BINARY_DIR}/src ${GSL_LDFLAGS} ${PYTHON_LDFLAGS}")

# Gathering of all flags from dependancies flags
# (e.g. for compiling libraries)
SET(PROJECT_ALL_DEP_LDFLAGS "${PROJECT_LDFLAGS} ${GSL_LDFLAGS} ${PYTHON_LDFLAGS}")

###################################
# Some prefix definitions
###################################

# Define the prefix where libs are going to be installed
SET(LIB_SUFFIX "" CACHE STRING "Define suffix of directory name (32/64)" )
SET(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}" CACHE STRING "Library directory name")

# Define the path to install the headers
SET(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include/${CMAKE_PROJECT_NAME}/" CACHE PATH "The subdirectory to the header prefix")

# Define the path to install the documentation
SET(DOC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/share/doc/${CMAKE_PROJECT_NAME}" CACHE PATH "The subdirectory to the documentation prefix")

# Define the path to install the pc pkg-config file
SET(PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}/pkgconfig/" CACHE STRING "Base directory for pkgconfig files")

###################################
# Pkg-config stuff
###################################

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
"
Name: ${CMAKE_PROJECT_NAME}
Description: ${PROJECT_DESCRIPTION_SUMMARY}
Version: ${PACKAGE_VERSION}
Requires: ${PKG_CONFIG_DEPENDS}
Libs: -L${LIB_INSTALL_DIR} ${PROJECT_LDFLAGS} ${PROJECT_LIBS}
Cflags: -I${INCLUDE_INSTALL_DIR} ${PROJECT_CFLAGS}
"
)


INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}.pc
    	DESTINATION ${PKGCONFIG_INSTALL_DIR}
	COMPONENT devel)

###################################
#  Subdirectories
###################################
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(doc)


###################################
#  Custom command for source tgz
###################################

SET(DIST_DIR "${CMAKE_PROJECT_NAME}-${PACKAGE_VERSION}")
ADD_CUSTOM_TARGET(dist 
		  COMMAND rm -rf ${DIST_DIR}
		  COMMAND mkdir  ${DIST_DIR}
		  COMMAND cp -r ${CMAKE_SOURCE_DIR}/* ${DIST_DIR} || true 
		  COMMAND rm -rf ${DIST_DIR}/build
		  COMMAND mkdir ${DIST_DIR}/build
  		  COMMAND tar --exclude="*~" --exclude="._*" --exclude=".svn" -zcvf ${DIST_DIR}.tar.gz ${DIST_DIR}
		  COMMAND rm -rf  ${DIST_DIR}
    	          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

###################################
# For building packages
###################################

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY ${PROJECT_DESCRIPTION_SUMMARY})
SET(CPACK_PACKAGE_VENDOR "Jeremy Fix")
SET(CPACK_PACKAGE_CONTACT "Jeremy.Fix@Supelec.fr")
SET(CPACK_PACKAGE_VERSION ${PACKAGE_VERSION})
SET(CPACK_PACKAGE_LICENSE "${PACKAGE_LICENSE}")
SET(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_SOURCE_DIR}/LICENSE)
SET(CPACK_RESOURCE_FILE_README ${CMAKE_SOURCE_DIR}/README)
SET(CPACK_RESOURCE_FILE_WELCOME ${CMAKE_SOURCE_DIR}/README)
SET(CPACK_RESOURCE_FILE_INSTALL ${CMAKE_SOURCE_DIR}/INSTALL)

IF(MYBUILDTYPE STREQUAL "binary")
   SET(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${CMAKE_PROJECT_NAME}-binaries;binary;/")
ELSEIF(MYBUILDTYPE STREQUAL "devel")
   SET(CPACK_INSTALL_CMAKE_PROJECTS "${CMAKE_BINARY_DIR};${CMAKE_PROJECT_NAME}-headers;devel;/")
ENDIF()

# Custom target for building the packages
# we do it inside 
execute_process(COMMAND getconf _NPROCESSORS_ONLN
                OUTPUT_VARIABLE NB_PROCESSOR)
ADD_CUSTOM_TARGET(packages
		  COMMAND rm -rf build
		  COMMAND mkdir build
		  COMMAND cd build && cmake ${CMAKE_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=/usr -DLIB_SUFFIX=${LIB_SUFFIX} -DMYBUILDTYPE="binary" && make package -j${NB_PROCESSOR}
		  COMMAND cp build/*.rpm build/*.deb . || true
		  COMMAND rm -rf build
		  COMMAND mkdir build
		  COMMAND cd build && cmake ${CMAKE_SOURCE_DIR} -DCMAKE_INSTALL_PREFIX=/usr -DLIB_SUFFIX=${LIB_SUFFIX} -DMYBUILDTYPE="devel" && make package -j${NB_PROCESSOR}
		  COMMAND cp build/*.rpm build/*.deb . || true
		  COMMAND rm -rf build
                  WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

###############################################
# For RPMs
###############################################

find_program( RPMBUILD
	 NAMES rpmbuild
	 PATHS "/usr/bin"
)

IF(NOT RPMBUILD STREQUAL "RPMBUILD-NOTFOUND")
  MESSAGE("Set up for building RPM")

  IF(MYBUILDTYPE STREQUAL "binary")
    SET(CPACK_RPM_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  ENDIF()

  IF(MYBUILDTYPE STREQUAL "binary")
    SET(CPACK_RPM_PACKAGE_REQUIRES "gsl, python2.7")
  ENDIF()

  IF(MYBUILDTYPE STREQUAL "devel")
    SET(CPACK_RPM_PACKAGE_NAME ${CMAKE_PROJECT_NAME}-devel)
    SET(CPACK_RPM_PACKAGE_REQUIRES "gsl-devel, python2.7-devel, ${CMAKE_PROJECT_NAME}")
  ENDIF()

  SET(CPACK_PACKAGE_FILE_NAME ${CPACK_RPM_PACKAGE_NAME}-${PACKAGE_VERSION}-${CMAKE_SYSTEM_PROCESSOR})

  SET(CPACK_RPM_SPEC_MORE_DEFINE "%define ignore \#")
  SET(CPACK_RPM_USER_FILELIST "%ignore /usr" "%ignore /usr/bin" "%ignore /usr/lib64" "%ignore /usr/lib" "%ignore /usr/lib64/pkgconfig" "%ignore /usr/lib/pkgconfig" "%ignore /usr/include" "%ignore /usr/share" "%ignore /usr/share/doc" "%ignore /usr/share/pixmaps" "%ignore /usr/share/applications" "%ignore /usr/lib/gstreamer-0.10" "%ignore /usr/lib64/gstreamer-0.10")

  SET(CPACK_RPM_PACKAGE_LICENSE "${PACKAGE_LICENSE}")
  SET(CPACK_GENERATOR "RPM")
ENDIF(NOT RPMBUILD STREQUAL "RPMBUILD-NOTFOUND")

###############################################
# For DEBs
###############################################
find_program( DPKGDEB
	 NAMES dpkg-deb
	 PATHS "/usr/bin"
)
IF(NOT DPKGDEB STREQUAL  "DPKGDEB-NOTFOUND")
  MESSAGE("Set up for building DEB")

  IF(MYBUILDTYPE STREQUAL "binary")
    SET(CPACK_DEBIAN_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
  ENDIF()

  IF(MYBUILDTYPE STREQUAL "binary")
    SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libgsl0ldbl, libpython2.7")
  ENDIF()

  IF(MYBUILDTYPE STREQUAL "devel")
    SET(CPACK_DEBIAN_PACKAGE_NAME ${CMAKE_PROJECT_NAME}-dev)
    SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libgsl0-dev, libpython2.7-dev, ${CMAKE_PROJECT_NAME}")
  ENDIF()

  SET(CPACK_PACKAGE_FILE_NAME ${CPACK_DEBIAN_PACKAGE_NAME}-${PACKAGE_VERSION}-${CMAKE_SYSTEM_PROCESSOR})

  SET(CPACK_DEBIAN_PACKAGE_MAINTAINER ${CPACK_PACKAGE_CONTACT})
  SET(CPACK_GENERATOR "DEB")
ENDIF(NOT DPKGDEB STREQUAL  "DPKGDEB-NOTFOUND")

###############################################
INCLUDE(CPack)
