# Create header-only target All other target will depend on it.
add_library(${PROJECT_NAME}_header_only INTERFACE ${${PROJECT_NAME}_HEADERS})

if(INITIALIZE_WITH_NAN)
    target_compile_definitions(${PROJECT_NAME}_header_only INTERFACE EIGEN_INITIALIZE_MATRICES_BY_NAN)
endif()

if(CHECK_RUNTIME_MALLOC)
    target_compile_definitions(${PROJECT_NAME}_header_only INTERFACE SIMPLEX_EIGEN_CHECK_MALLOC
                                                                    EIGEN_RUNTIME_NO_MALLOC)
endif(CHECK_RUNTIME_MALLOC)
if(ENABLE_TEMPLATE_INSTANTIATION)
    target_compile_definitions(${PROJECT_NAME}_header_only
                                INTERFACE SIMPLEX_ENABLE_TEMPLATE_INSTANTIATION)
endif()


target_link_libraries(${PROJECT_NAME}_header_only INTERFACE Eigen3::Eigen pinocchio::pinocchio
                                                            Boost::boost Boost::serialization)

# add precompiled header for pinocchio and eigen headers
if(USE_PRECOMPILED_HEADERS)
    target_precompile_headers(${PROJECT_NAME}_header_only INTERFACE
                                ${PROJECT_SOURCE_DIR}/include/simplex/pch.hpp)
endif()

# Enable diffcoal for collision detection derivatives
if(SIMPLEX_USE_DIFFCOAL)
    target_link_libraries(${PROJECT_NAME}_header_only INTERFACE diffcoal::diffcoal)
else(SIMPLEX_USE_DIFFCOAL)
    target_compile_definitions(${PROJECT_NAME}_header_only
                                INTERFACE SIMPLEX_SKIP_COLLISION_DERIVATIVES_CONTRIBUTIONS)
endif(SIMPLEX_USE_DIFFCOAL)

# Enable Clarabel constraint solver
if(SIMPLEX_USE_CLARABEL)
	if(TARGET clarabel::clarabel)
		target_link_libraries(${PROJECT_NAME}_header_only INTERFACE clarabel::clarabel)
	elseif(TARGET PkgConfig::clarabel)
		target_link_libraries(${PROJECT_NAME}_header_only INTERFACE PkgConfig::clarabel)
	else()
		# Fallback: Try to find the library directly in the conda environment
		find_library(
		CLARABEL_LIBRARY
		NAMES clarabel_c
		PATHS ${CMAKE_INSTALL_PREFIX}/lib ENV LD_LIBRARY_PATH)
		if(CLARABEL_LIBRARY)
			message(STATUS "Found Clarabel library at: ${CLARABEL_LIBRARY}")
			target_link_libraries(${PROJECT_NAME}_header_only INTERFACE ${CLARABEL_LIBRARY})
		else()
			message(WARNING "Clarabel library not found. Please install Clarabel.cpp")
		endif()
	endif()
endif(SIMPLEX_USE_CLARABEL)

# Enable introspection of private/protected members and methods Can be useful for core developpers
# when trying to debug stuff
if(SIMPLEX_ENABLE_PRIVATE_INTROSPECTION)
	target_compile_definitions(${PROJECT_NAME}_header_only
								INTERFACE SIMPLEX_ENABLE_PRIVATE_INTROSPECTION)
endif()

# Enable tracy profiling
if(SIMPLEX_TRACY_ENABLE)
    modernize_target_link_libraries(
        ${PROJECT_NAME}_header_only
        SCOPE INTERFACE
        TARGETS Tracy::TracyClient)
endif()

target_include_directories(${PROJECT_NAME}_header_only INTERFACE $<INSTALL_INTERFACE:include>)

cxx_flags_by_compiler_frontend(MSVC "/bigobj" OUTPUT PUBLIC_OPTIONS FILTER)
target_compile_options(${PROJECT_NAME}_header_only INTERFACE ${PUBLIC_OPTIONS})
cxx_flags_by_compiler_frontend(MSVC "NOMINMAX" OUTPUT PUBLIC_DEFINITIONS)
target_compile_definitions(${PROJECT_NAME}_header_only INTERFACE ${PUBLIC_DEFINITIONS})

add_header_group(${PROJECT_NAME}_CORE_PUBLIC_HEADERS)
add_header_group(${PROJECT_NAME}_CORE_GENERATED_PUBLIC_HEADERS)

install(
    TARGETS ${PROJECT_NAME}_header_only
    EXPORT ${TARGETS_EXPORT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(ENABLE_TEMPLATE_INSTANTIATION)
    add_library(
        ${PROJECT_NAME}_pinocchio_ti SHARED ${${PROJECT_NAME}_PINOCCHIO_TEMPLATE_INSTANTIATION_SOURCES}
                                            ${${PROJECT_NAME}_PINOCCHIO_TEMPLATE_INSTANTIATION_HEADERS})
    target_link_libraries(${PROJECT_NAME}_pinocchio_ti PUBLIC ${PROJECT_NAME}_header_only)

    install(
        TARGETS ${PROJECT_NAME}_pinocchio_ti
        EXPORT ${TARGETS_EXPORT_NAME}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_CORE_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_header_only)

if(ENABLE_TEMPLATE_INSTANTIATION)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_pinocchio_ti)
endif()

# Install main library
install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)