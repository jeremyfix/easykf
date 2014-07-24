FIND_PACKAGE(Doxygen)
IF(NOT DOXYGEN_FOUND)
    message("Doxygen not found, I will not generate/install the documentation")
ELSE()
   configure_file(Doxyfile.in Doxyfile)

   SET(DOXYGEN_INPUT ${CMAKE_BINARY_DIR}/doc/Doxyfile)
   SET(APIDOC_DIR ${CMAKE_BINARY_DIR}/doc CACHE PATH "API documentation directory")
   SET(DOXYGEN_OUTPUT ${APIDOC_DIR}/html/index.html)

   ADD_CUSTOM_COMMAND(
     OUTPUT ${DOXYGEN_OUTPUT}
     COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
     COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
     COMMAND ${CMAKE_COMMAND} -E echo "Done."
     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
     DEPENDS ${DOXYGEN_INPUT}
  )

  ADD_CUSTOM_TARGET(apidoc ALL DEPENDS ${DOXYGEN_OUTPUT})

  INSTALL(DIRECTORY ${APIDOC_DIR}/html
          DESTINATION share/doc/easykf/)


ENDIF()

