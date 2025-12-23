# Copy all files in this directory to simplex/test/resources/... in the build directory
file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SRC_FILE ${SRC_FILES})
    file(RELATIVE_PATH RELATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${SRC_FILE})
    configure_file(
        ${SRC_FILE}
        ${CMAKE_BINARY_DIR}/SIMPLEX/test/resources/${RELATIVE_PATH}
        COPYONLY
    )
endforeach()