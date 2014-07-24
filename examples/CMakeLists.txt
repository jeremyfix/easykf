# Make sure the compiler can find include files from our library.
include_directories (${CMAKE_SOURCE_DIR}/src)

# Define the libs used to compile the examples
set(LIBS_FOR_EXAMPLE easykf ${GSL_LIBRARIES})

# Define our examples to compile
set(USAGE_EXAMPLES example-001 example-002
		   example-003 example-004
		   example-005 example-006
		   example-007 example-008
		   example-009)

# loop over the list to compile them
foreach(f ${USAGE_EXAMPLES})
    add_executable (${f} ${f}.cc)
    target_link_libraries (${f} ${LIBS_FOR_EXAMPLE})	
endforeach(f) 

foreach(f ${USAGE_EXAMPLES})
    install(PROGRAMS ${CMAKE_BINARY_DIR}/examples/${f} 
            DESTINATION bin
            RENAME easykf-${f})
endforeach(f) 
