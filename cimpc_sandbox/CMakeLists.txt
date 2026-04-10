cmake_minimum_required(VERSION 3.22)
project(cimpc_sandbox)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Find Boost.UnitTestFramework
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

# Find Conda installed dependencies based on your list
find_package(Eigen3 REQUIRED)
find_package(hpp-fcl CONFIG REQUIRED)
find_package(cereal CONFIG REQUIRED)
find_package(proxsuite CONFIG REQUIRED)
find_package(OpenMP REQUIRED)
# find make install package
find_package(pinocchio CONFIG REQUIRED)
find_package(crocoddyl CONFIG REQUIRED)

# CPP tests in tests/ directory
file(GLOB_RECURSE CPP_TEST_FILES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")
foreach(TEST_FILE ${CPP_TEST_FILES})
	# Get the name without extension
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
	
	# Compute a unique test executable name
	string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" REL_PATH "${TEST_FILE}")
	string(REGEX REPLACE "[/]" "-" EXEC_NAME "${REL_PATH}")
	string(REPLACE ".cpp" "" EXEC_NAME "${EXEC_NAME}")
	set(FULL_TEST_NAME "test-cpp-${EXEC_NAME}")

	add_executable(${FULL_TEST_NAME} ${TEST_FILE})
	add_test(NAME ${FULL_TEST_NAME} COMMAND ${FULL_TEST_NAME})

	set(MODULE_NAME "Test_${TEST_NAME}")
	string(REPLACE "-" "_" MODULE_NAME ${MODULE_NAME})
	target_compile_definitions(
		${FULL_TEST_NAME} PRIVATE BOOST_TEST_DYN_LINK
							BOOST_TEST_MODULE=${MODULE_NAME})

	# Copy executable to build directory root instead of nested
	set_target_properties(${FULL_TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

	# There is no RPATH in Windows, then we must use the PATH to find the DLL
	if(WIN32)
		string(REPLACE ";" "\\\;" _PATH "$ENV{PATH}")
		set_tests_properties(${FULL_TEST_NAME} PROPERTIES ENVIRONMENT "PATH=${_PATH}")
	endif()

	target_include_directories(${FULL_TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
	
	# Link newly added conda dependencies
	target_link_libraries(${FULL_TEST_NAME} PRIVATE 
		Boost::unit_test_framework
		Eigen3::Eigen
		hpp-fcl::hpp-fcl
		cereal::cereal
		proxsuite::proxsuite
		pinocchio::pinocchio
		crocoddyl::crocoddyl
		contactbench
	)
endforeach()

