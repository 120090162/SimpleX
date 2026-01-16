cmake_minimum_required(VERSION 3.22)

set(PROJECT_NAME simplex)
set(PROJECT_DESCRIPTION "The Simple-X: a Simulation based on Simple")
set(PROJECT_URL "https://github.com/120090162/SimpleX")
set(PROJECT_CUSTOM_HEADER_EXTENSION "hpp")
set(PROJECT_USE_CMAKE_EXPORT TRUE)
set(PROJECT_USE_KEYWORD_LINK_LIBRARIES TRUE)
set(PROJECT_COMPATIBILITY_VERSION AnyNewerVersion)
set(SIMPLEX_PROJECT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR})

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(CMAKE_VERBOSE_MAKEFILE True)
# Export compile commands for clang-tidy
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Allow us to find resources (e.g. urdfs/xmls) in the build directory.
add_compile_definitions(SIMPLEX_TEST_DIR="${CMAKE_SOURCE_DIR}")

# ref to https://github.com/xizhibei/blog/issues/145
# 使用ccache提高编译速度
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "Set up ccache ...")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else()
	message(STATUS "ccache not found.")
endif()

# 开启 Interprocedural optimization
include(CheckIPOSupported)
check_ipo_supported(RESULT _IsIPOSupported)
if(_IsIPOSupported)
    message(STATUS "Turn on INTERPROCEDURAL_OPTIMIZATION")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
	message(STATUS "Turn off INTERPROCEDURAL_OPTIMIZATION.")
endif()

# ----------------------------------------------------
# --- OPTIONS  ---------------------------------------
# ----------------------------------------------------

# Need to be set before including base.cmake
option(INSTALL_DOCUMENTATION "Generate and install the documentation" OFF)
option(USE_PRECOMPILED_HEADERS "Use pre-compiled headers for pinocchio and eigen." OFF)

# Check if the submodule cmake have been initialized
set(JRL_CMAKE_MODULES "${CMAKE_CURRENT_LIST_DIR}/cmake")
if(EXISTS "${JRL_CMAKE_MODULES}/base.cmake")
	message(STATUS "JRL cmakemodules found in 'cmake/' git submodule")
else()
	find_package(jrl-cmakemodules QUIET CONFIG)
	if(jrl-cmakemodules_FOUND)
		get_property(
		JRL_CMAKE_MODULES
		TARGET jrl-cmakemodules::jrl-cmakemodules
		PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
		message(STATUS "JRL cmakemodules found on system at ${JRL_CMAKE_MODULES}")
	elseif(${CMAKE_VERSION} VERSION_LESS "3.14.0")
		message(
		FATAL_ERROR
			"\nCan't find jrl-cmakemodules. Please either:\n"
			"  - use git submodule: 'git submodule update --init'\n"
			"  - or install https://github.com/jrl-umi3218/jrl-cmakemodules\n"
			"  - or upgrade your CMake version to >= 3.14 to allow automatic fetching\n")
	else()
		message(STATUS "JRL cmakemodules not found. Let's fetch it.")
		include(FetchContent)
		FetchContent_Declare(
		"jrl-cmakemodules"
		GIT_REPOSITORY "https://github.com/jrl-umi3218/jrl-cmakemodules.git"
		EXCLUDE_FROM_ALL)
		FetchContent_MakeAvailable("jrl-cmakemodules")
		FetchContent_GetProperties("jrl-cmakemodules" SOURCE_DIR JRL_CMAKE_MODULES)
	endif()
endif()

set(DOXYGEN_USE_MATHJAX YES)
set(DOXYGEN_USE_TEMPLATE_CSS YES)

if(POLICY CMP0177)
	cmake_policy(SET CMP0177 NEW)
	set(CMAKE_POLICY_DEFAULT_CMP0177 NEW)
endif()
include("${JRL_CMAKE_MODULES}/base.cmake")

compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})

include("${JRL_CMAKE_MODULES}/tracy.cmake")
include("${JRL_CMAKE_MODULES}/boost.cmake")
include("${JRL_CMAKE_MODULES}/ide.cmake")
include("${JRL_CMAKE_MODULES}/apple.cmake")
if(APPLE) # Use the handmade approach
	if(${CMAKE_VERSION} VERSION_LESS "3.18.0") # Need to find the right version
		set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
	endif()
elseif(UNIX)
	if(${CMAKE_VERSION} VERSION_EQUAL "3.20.0")
		set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/find-external/OpenMP ${CMAKE_MODULE_PATH})
	endif()
endif()
include(CMakeDependentOption)

# If needed, set CMake policy for APPLE systems
apply_default_apple_configuration()
if(CMAKE_VERSION VERSION_GREATER "3.12")
	cmake_policy(SET CMP0074 NEW)
endif()

# Force C++ standard to be C++14 at least
check_minimal_cxx_standard(14 ENFORCE)

option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
option(BUILD_TEST_CASES "Build the simulator tests" OFF)
option(BUILD_PYTHON_INTERFACE "Build the Python bindings" ON)
option(GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" OFF)
option(BUILD_WITH_COMMIT_VERSION "Build libraries by setting specific commit version" OFF)
option(SIMPLEX_USE_DIFFCOAL "Use diffcoal to compute collision detection derivatives" OFF)
option(SIMPLEX_USE_CLARABEL "Use Clarabel as a constraint solver option" ON)
option(SIMPLEX_ENABLE_PRIVATE_INTROSPECTION
       "Turns off private/protected members and methods in classes, to allow for easier debugging"
       ON)

option(BUILD_WITH_OPENMP_SUPPORT "Build the library with the OpenMP support" OFF)
if(APPLE)
	option(BUILD_WITH_ACCELERATE_SUPPORT "Build EigenPy with the Accelerate support" OFF)
else(APPLE)
	set(BUILD_WITH_ACCELERATE_SUPPORT FALSE)
endif(APPLE)

option(INITIALIZE_WITH_NAN "Initialize Eigen entries with NaN" OFF)
option(CHECK_RUNTIME_MALLOC "Check if some memory allocations are performed at runtime" OFF)
option(ENABLE_TEMPLATE_INSTANTIATION "Template instantiation of the main library" ON)
option(SIMPLEX_DOWNLOAD_TRACY "Use FetchContent to install Tracy." OFF)

set(CFLAGS_OPTIONS)

if(INITIALIZE_WITH_NAN)
	message(STATUS "Initialize with NaN all the Eigen entries.")
endif(INITIALIZE_WITH_NAN)

if(CHECK_RUNTIME_MALLOC)
	message(STATUS "Check if some memory allocations are performed at runtime.")
endif(CHECK_RUNTIME_MALLOC)

if(ENABLE_TEMPLATE_INSTANTIATION)
	message(STATUS "Template instantiation of the main library")
	list(APPEND CFLAGS_OPTIONS "-DSIMPLEX_ENABLE_TEMPLATE_INSTANTIATION")
endif(ENABLE_TEMPLATE_INSTANTIATION)

macro(TAG_LIBRARY_VERSION target)
	set_target_properties(${target} PROPERTIES SOVERSION ${PROJECT_VERSION})
endmacro(TAG_LIBRARY_VERSION)

# ----------------------------------------------------
# --- DEPENDENCIES -----------------------------------
# ----------------------------------------------------

# -- Eigen
add_project_dependency(Eigen3 REQUIRED PKG_CONFIG_REQUIRES "eigen3 >= 3.0.5")
if(SIMPLEX_USE_DIFFCOAL)
	add_project_dependency(diffcoal REQUIRED PKG_CONFIG_REQUIRES)
endif(SIMPLEX_USE_DIFFCOAL)

# -- boost
set(BOOST_REQUIRED_COMPONENTS filesystem serialization system)
set_boost_default_options()
export_boost_default_options()
add_project_dependency(Boost REQUIRED COMPONENTS ${BOOST_REQUIRED_COMPONENTS})

# -- pinocchio
add_project_dependency(pinocchio REQUIRED PKG_CONFIG_REQUIRES "pinocchio >= 2.9.2")

# -- clarabel (optional)
if(SIMPLEX_USE_CLARABEL)
	message(STATUS "Building with Clarabel constraint solver support")
	find_package(clarabel QUIET)
	if(NOT clarabel_FOUND)
		message(STATUS "Clarabel not found via find_package, trying pkg-config")
		find_package(PkgConfig QUIET)
		if(PKG_CONFIG_FOUND)
		pkg_check_modules(clarabel QUIET IMPORTED_TARGET clarabel)
		endif()
	endif()

	if(clarabel_FOUND OR TARGET PkgConfig::clarabel)
		message(STATUS "Clarabel found and will be linked")
		add_definitions(-DSIMPLEX_WITH_CLARABEL_SUPPORT)
	else()
		message(
		WARNING
			"Clarabel support requested but library not found. Please install Clarabel.cpp or set SIMPLEX_USE_CLARABEL=OFF"
		)
	endif()
endif(SIMPLEX_USE_CLARABEL)

# -- tracy (optional)
if(SIMPLEX_TRACY_ENABLE AND SIMPLEX_DOWNLOAD_TRACY)
	# We use FetchContent_Populate because we need EXCLUDE_FROM_ALL to avoid installing Tracy with
	# simplex. We can directly use EXCLUDE_FROM_ALL in FetchContent_Declare when CMake minimum version
	# will be 3.28.
	if(POLICY CMP0169)
		cmake_policy(SET CMP0169 OLD)
	endif()
	FetchContent_Declare(
		tracy
		GIT_REPOSITORY https://github.com/Simple-Robotics/tracy.git
		GIT_TAG patches
		GIT_SHALLOW TRUE
		GIT_PROGRESS TRUE)
	FetchContent_GetProperties(tracy)
	if(NOT tracy_POPULATED)
		FetchContent_Populate(tracy)
		set(TRACY_STATIC
			ON
			CACHE INTERNAL "")
		set(TRACY_ENABLE
			${SIMPLEX_TRACY_ENABLE}
			CACHE INTERNAL "")
		add_subdirectory(${tracy_SOURCE_DIR} ${tracy_BINARY_DIR} EXCLUDE_FROM_ALL)
		# Extract the target include directories, set as system
		get_target_property(tracy_INCLUDE_DIR TracyClient INTERFACE_INCLUDE_DIRECTORIES)
		set_target_properties(
		TracyClient PROPERTIES POSITION_INDEPENDENT_CODE True INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
																"${tracy_INCLUDE_DIR}")
	endif()
elseif(SIMPLEX_TRACY_ENABLE)
	# assume it is installed somewhere
	add_project_dependency(Tracy REQUIRED)
	if(NOT ${tracy_FOUND})
		message(
		FATAL_ERROR "SimpleX support for tracy is enabled, but tracy was not found on your system."
					" Install it, or set the option SIMPLEX_DOWNLOAD_TRACY to ON so we can fetch it.")
	else()
		message(STATUS "Tracy found on your system at ${Tracy_DIR}")
	endif()
endif()

# -- python
if(BUILD_PYTHON_INTERFACE)
	message(
		STATUS
		"The Python bindings of SimpleX will be compiled along the main library. If you want to disable this feature, please set the option BUILD_PYTHON_INTERFACE to OFF."
	)

	set(PYTHON_COMPONENTS Interpreter Development NumPy)
	add_project_dependency(eigenpy 2.7.10 REQUIRED)

	# Check wether this a PyPy Python
	execute_process(
		COMMAND ${PYTHON_EXECUTABLE} -c "import platform; print(platform.python_implementation())"
		OUTPUT_VARIABLE _python_implementation_value
		OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

	message(STATUS "Python compiler: ${_python_implementation_value}")
	if(_python_implementation_value MATCHES "PyPy")
		set(BUILD_PYTHON_INTERFACE_WITH_PYPY TRUE)
	endif()

else(BUILD_PYTHON_INTERFACE)
	message(
		STATUS
		"SimpleX won't be compiled with its Python bindings. If you want to enable this feature, please set the option BUILD_PYTHON_INTERFACE to ON."
	)
endif(BUILD_PYTHON_INTERFACE)

if(BUILD_WITH_OPENMP_SUPPORT)
	add_project_dependency(OpenMP REQUIRED)
endif()

if(BUILD_WITH_ACCELERATE_SUPPORT)
	if(NOT ${Eigen3_VERSION} VERSION_GREATER_EQUAL "3.4.90")
		message(
		FATAL_ERROR
			"Your version of Eigen is too low. Should be at least 3.4.90. Current version is ${Eigen3_VERSION}."
		)
	endif()

	set(CMAKE_MODULE_PATH ${JRL_CMAKE_MODULES}/find-external/Accelerate ${CMAKE_MODULE_PATH})
	# FIND_EXTERNAL "Accelerate".
	#
	# We don't export yet as there might be an issue on AMR64 platforms.
	find_package(Accelerate REQUIRED)
	message(STATUS "Build with Accelerate support framework.")
	add_definitions(-DPINOCCHIO_WITH_ACCELERATE_SUPPORT -DSIMPLEX_WITH_ACCELERATE_SUPPORT)
endif(BUILD_WITH_ACCELERATE_SUPPORT)

# ----------------------------------------------------
# --- MAIN LIBRARY -----------------------------------
# ----------------------------------------------------
# Sources definition
include(sources.cmake)

# List headers to install
list(APPEND ${PROJECT_NAME}_HEADERS ${${PROJECT_NAME}_CORE_PUBLIC_HEADERS})

if(ENABLE_TEMPLATE_INSTANTIATION)
	list(APPEND ${PROJECT_NAME}_CORE_SOURCES ${${PROJECT_NAME}_TEMPLATE_INSTANTIATION_SOURCES})
	list(APPEND ${PROJECT_NAME}_CORE_PUBLIC_HEADERS
		${${PROJECT_NAME}_TEMPLATE_INSTANTIATION_PUBLIC_HEADERS})
	list(APPEND ${PROJECT_NAME}_HEADERS ${${PROJECT_NAME}_TEMPLATE_INSTANTIATION_PUBLIC_HEADERS})
	list(APPEND ${PROJECT_NAME}_HEADERS ${${PROJECT_NAME}_PINOCCHIO_TEMPLATE_INSTANTIATION_HEADERS})
endif()

if(BUILD_PYTHON_INTERFACE)
	list(APPEND ${PROJECT_NAME}_HEADERS ${${PROJECT_NAME}_BINDINGS_PYTHON_PUBLIC_HEADERS})
endif()
add_subdirectory(src)

# ----------------------------------------------------
# --- BENCHMARK --------------------------------------
# ----------------------------------------------------
if(BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

# ----------------------------------------------------
# --- BINDINGS ---------------------------------------
# ----------------------------------------------------
if(BUILD_PYTHON_INTERFACE)
	add_subdirectory(bindings)
endif()

# ----------------------------------------------------
# --- UNIT TESTS -------------------------------------
# ----------------------------------------------------
# test data
if(BUILD_TEST_CASES)
	add_subdirectory(tests)
endif(BUILD_TEST_CASES)

# Export for PkgConfig
pkg_config_append_libs(${PROJECT_NAME})
pkg_config_append_boost_libs(${BOOST_COMPONENTS})
setup_project_finalize()