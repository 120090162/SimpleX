include(${JRL_CMAKE_MODULES}/python-helpers.cmake)
include(${JRL_CMAKE_MODULES}/stubs.cmake)

# --- PYTHON TARGET --- #
set(PYWRAP ${PROJECT_NAME}_pywrap)
set(PYWRAP
    ${PYWRAP}
    PARENT_SCOPE)

# --- COMPILE WRAPPER
make_directory("${${PROJECT_NAME}_BINARY_DIR}/bindings/python/${PROJECT_NAME}")

set(${PYWRAP}_SOURCES ${${PROJECT_NAME}_BINDINGS_PYTHON_SOURCES})
set(${PYWRAP}_HEADERS ${${PROJECT_NAME}_BINDINGS_PYTHON_PUBLIC_HEADERS})

if(BUILD_PYTHON_INTERFACE)
	add_custom_target(python)
	set_target_properties(python PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD True)
	python_build_get_target(python_build_target)
	add_dependencies(python ${python_build_target})

	set(PKG_CONFIG_PYWRAP_REQUIRES "eigenpy >= 2.6.5")
	if(IS_ABSOLUTE ${PYTHON_SITELIB})
		set(ABSOLUTE_PYTHON_SITELIB ${PYTHON_SITELIB})
	else()
		set(ABSOLUTE_PYTHON_SITELIB ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB})
	endif()
	set(SIMPLEX_PYTHON_INSTALL_DIR ${ABSOLUTE_PYTHON_SITELIB}/${PROJECT_NAME})

	function(INSTALL_PYTHON_FILES)
		set(options)
		set(oneValueArgs MODULE)
		set(multiValueArgs FILES)
		cmake_parse_arguments(ARGS "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

		set(SOURCE_PATH ${PROJECT_NAME})
		set(INSTALL_PATH ${SIMPLEX_PYTHON_INSTALL_DIR})
		if(ARGS_MODULE)
			set(SOURCE_PATH ${SOURCE_PATH}/${ARGS_MODULE})
			set(INSTALL_PATH ${INSTALL_PATH}/${ARGS_MODULE})
		endif()

		foreach(f ${ARGS_FILES})
			python_build(${SOURCE_PATH} ${f})
			install(FILES ${SOURCE_PATH}/${f} DESTINATION ${INSTALL_PATH})
		endforeach()
	endfunction()

	# --- PYTHON LIB --- #
	set(PYTHON_LIB_NAME "${PYWRAP}")
	set(${PYTHON_LIB_NAME}_SOURCES ${${PYWRAP}_SOURCES})
	set(${PYTHON_LIB_NAME}_HEADERS ${${PYWRAP}_HEADERS})

	add_library(${PYTHON_LIB_NAME} SHARED ${${PYTHON_LIB_NAME}_SOURCES} ${${PYTHON_LIB_NAME}_HEADERS})

	add_dependencies(python ${PYTHON_LIB_NAME})
	set_target_properties(${PYTHON_LIB_NAME} PROPERTIES PREFIX "") # Remove lib prefix for the target

	# Do not report: -Wconversion as the BOOST_PYTHON_FUNCTION_OVERLOADS implicitly converts.
	# -Wcomment as latex equations have multi-line comments. -Wself-assign-overloaded as bp::self
	# operations trigger this
	if(NOT WIN32)
		target_compile_options(${PYTHON_LIB_NAME} PRIVATE -Wno-conversion -Wno-comment
														-Wno-self-assign-overloaded)
	endif(NOT WIN32)

	set_target_properties(${PYTHON_LIB_NAME} PROPERTIES VERSION ${PROJECT_VERSION})
	add_header_group(${PYTHON_LIB_NAME}_HEADERS)
	add_source_group(${PYTHON_LIB_NAME}_SOURCES)

	modernize_target_link_libraries(
		${PYTHON_LIB_NAME}
		SCOPE PUBLIC
		TARGETS eigenpy::eigenpy)
	target_link_libraries(${PYTHON_LIB_NAME} PUBLIC ${PROJECT_NAME})
	target_compile_definitions(${PYTHON_LIB_NAME}
								PRIVATE -DPINOCCHIO_PYTHON_INTERFACE_WITH_HPP_FCL_PYTHON_BINDINGS)
	if(WIN32)
		target_compile_definitions(${PYTHON_LIB_NAME} PRIVATE -DNOMINMAX)
		target_link_libraries(${PYTHON_LIB_NAME} PUBLIC ${PYTHON_LIBRARY})
	endif(WIN32)
  	target_compile_options(${PYTHON_LIB_NAME} PRIVATE -DSIMPLEX_PYTHON_MODULE_NAME=${PYTHON_LIB_NAME})

  	set(${PYWRAP}_INSTALL_DIR ${ABSOLUTE_PYTHON_SITELIB}/${PROJECT_NAME})

	set_target_properties(
		${PYTHON_LIB_NAME}
		PROPERTIES PREFIX ""
				SUFFIX ${PYTHON_EXT_SUFFIX}
				OUTPUT_NAME "${PYTHON_LIB_NAME}"
				LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bindings/python/${PROJECT_NAME}"
				# On Windows, shared library are treat as binary
				RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bindings/python/${PROJECT_NAME}")

	if(UNIX AND NOT APPLE)
		get_relative_rpath(${${PYWRAP}_INSTALL_DIR} ${PYWRAP}_INSTALL_RPATH)
		set_target_properties(${PYTHON_LIB_NAME} PROPERTIES INSTALL_RPATH "${${PYWRAP}_INSTALL_RPATH}")
	endif()

	install(
		TARGETS ${PYTHON_LIB_NAME}
		EXPORT ${TARGETS_EXPORT_NAME}
		DESTINATION ${SIMPLEX_PYTHON_INSTALL_DIR})

	install_python_files(FILES __init__.py)

	# --- STUBS --- #
	if(GENERATE_PYTHON_STUBS)
		python_build_get_target(python_build_target_name)
		load_stubgen()
		# Set PYWRAP and PROJECT_NAME as stubs dependencies.
		#
		# PROJECT_NAME is mandatory (even if it's a PYWRAP dependency) to find PROJECT_NAME name DLL on
		# windows.
		generate_stubs(
			${CMAKE_CURRENT_BINARY_DIR}
			${PROJECT_NAME}
			${ABSOLUTE_PYTHON_SITELIB}
			${PYWRAP}
			${PROJECT_NAME}
			${STUBGEN_DEPENDENCIES}
			${python_build_target_name})
	endif(GENERATE_PYTHON_STUBS)

endif(BUILD_PYTHON_INTERFACE)
