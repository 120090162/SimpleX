# Find Boost.UnitTestFramework
find_package(Boost COMPONENTS unit_test_framework)

# Compute flags outside the macro to avoid recomputing it for each tests
cxx_flags_by_compiler_frontend(MSVC _USE_MATH_DEFINES OUTPUT TEST_PRIVATE_DEFINITIONS)

set(SIMPLEX_UNIT_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(GET_CPP_TEST_NAME name src_dir full_test_name)
	string(REPLACE "${SIMPLEX_UNIT_TEST_DIR}" "" prefix_name "${src_dir}")
	string(REGEX REPLACE "[/]" "-" prefix_name "${prefix_name}-")
	set(${full_test_name}
		"${PROJECT_NAME}-test-cpp${prefix_name}${name}"
		PARENT_SCOPE)
endfunction()

function(ADD_SIMPLEX_UNIT_TEST name)
	set(oneValueArgs)
	set(multiValueArgs PACKAGES)
	cmake_parse_arguments(unit_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

	set(PKGS ${unit_test_PACKAGES})

	get_cpp_test_name(${name} ${CMAKE_CURRENT_SOURCE_DIR} TEST_NAME)
	add_unit_test(${TEST_NAME} ${name}.cpp)

	set(MODULE_NAME "Test")
	string(REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
	target_compile_definitions(
		${TEST_NAME} PRIVATE ${TEST_PRIVATE_DEFINITIONS} BOOST_TEST_DYN_LINK
							BOOST_TEST_MODULE=${MODULE_NAME} SIMPLEX_MODEL_DIR=\"${SIMPLEX_MODEL_DIR}\")

	# There is no RPATH in Windows, then we must use the PATH to find the DLL
	if(WIN32)
		string(REPLACE ";" "\\\;" _PATH "$ENV{PATH}")
		set(ENV_VARIABLES
			"PATH=${_PATH}\\\;${CMAKE_BINARY_DIR}/src\\\;${CMAKE_BINARY_DIR}/bindings/python/simplex")
		set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT "${ENV_VARIABLES}")
	endif()

	set_target_properties(${TEST_NAME} PROPERTIES LINKER_LANGUAGE CXX)
	target_include_directories(${TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	target_link_libraries(${TEST_NAME} PUBLIC ${PROJECT_NAME})
	# target_link_libraries(${TEST_NAME} PUBLIC Eigen3::Eigen)
	target_link_libraries(${TEST_NAME} PRIVATE Boost::unit_test_framework)

	if(PKGS)
		target_link_libraries(${TEST_NAME} PRIVATE ${PKGS})
	endif()
endfunction()

# CPP test
add_subdirectory(forward)
# add_subdirectory(resources)

# Python tests
if(BUILD_PYTHON_INTERFACE)
	file(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_UNITTEST *.py)

	foreach(TEST_FILE ${${PROJECT_NAME}_PYTHON_UNITTEST})
		get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
		string(REGEX REPLACE "${PROJECT_SOURCE_DIR}/" "" TEST_FILE ${TEST_FILE})
		add_python_unit_test("${PROJECT_NAME}-test-py-${TEST_NAME}" "${TEST_FILE}" "bindings")
	endforeach(TEST_FILE ${${PROJECT_NAME}_PYTHON_UNITTEST})
endif(BUILD_PYTHON_INTERFACE)
