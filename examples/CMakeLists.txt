# Make sure the compiler can find include files from our library.
include_directories (${CMAKE_SOURCE_DIR}/src)

# Define our examples to compile
set(USAGE_EXAMPLES example-001 example-002
		   example-003 example-004
		   example-005 example-006
		   example-007 example-008
		   example-009)

# loop over the list to compile them
foreach(f ${USAGE_EXAMPLES})
    add_executable (${f} ${f}.cc)
    set_target_properties(${f} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${f} PROPERTIES COMPILE_FLAGS "${PROJECT_ALL_CFLAGS}" LINK_FLAGS "${PROJECT_ALL_LDFLAGS}")

endforeach(f) 

foreach(f ${USAGE_EXAMPLES})
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${f} 
            DESTINATION bin
            RENAME ${CMAKE_PROJECT_NAME}-${f}
            COMPONENT binary)
endforeach(f) 
