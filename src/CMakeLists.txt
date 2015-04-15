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


# Installation targets
install (TARGETS easykf 
	 DESTINATION ${LIB_INSTALL_DIR})
install (FILES ${headers_files} 
	 DESTINATION ${INCLUDE_INSTALL_DIR})

# Define our library 
add_library(
	pyeasykf
	SHARED
	pyeasykf.cpp
)
target_link_libraries(pyeasykf ${Boost_LIBRARIES} ${PYTHON_LIBS} "-lboost_python" ${GSL_LIBRARIES})
