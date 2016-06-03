# Generate the list of source files
# using GLOB does not RECURSE
file(
	GLOB
	source_files
	*.h main.cpp
)

file(
	GLOB
	headers_files
	*.h
)

# Define our library 
add_library(
	easykf
	SHARED
	${source_files}
)
set_target_properties(easykf PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(easykf PROPERTIES COMPILE_FLAGS "${PROJECT_ALL_CFLAGS}" LINK_FLAGS "${PROJECT_ALL_DEP_LDFLAGS}")

# Installation targets
install (TARGETS easykf 
	 DESTINATION ${LIB_INSTALL_DIR}
     COMPONENT binary)
install (FILES ${headers_files} 
	 DESTINATION ${INCLUDE_INSTALL_DIR}
     COMPONENT devel)

# Define our library 
add_library(
	pyeasykf
	SHARED
	pyeasykf.cpp
)
set_target_properties(pyeasykf PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(pyeasykf PROPERTIES COMPILE_FLAGS "${PROJECT_ALL_CFLAGS}" LINK_FLAGS "${PROJECT_ALL_DEP_LDFLAGS} -lboost_python")

