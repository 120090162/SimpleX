# Define Simplex sources and headers

# --- Core library ---
set(${PROJECT_NAME}_CORE_SOURCES empty.cpp)

set(${PROJECT_NAME}_CORE_PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/include/simplex/core/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/core/contact-frame.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/core/contact-frame.hxx
    ${PROJECT_SOURCE_DIR}/include/simplex/core/constraints-problem.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/core/constraints-problem.hxx
    ${PROJECT_SOURCE_DIR}/include/simplex/core/diff-solver.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/core/simulator.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/core/simulator.hxx
    ${PROJECT_SOURCE_DIR}/include/simplex/math/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/math/qr.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/solver/clarabel-solver.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/solver/clarabel-solver.hxx
    ${PROJECT_SOURCE_DIR}/include/simplex/utils/visitors.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/macros.hpp)

set(_binary_headers_root ${${PROJECT_NAME}_BINARY_DIR}/include/simplex)
set(${PROJECT_NAME}_CORE_GENERATED_PUBLIC_HEADERS
    ${_binary_headers_root}/config.hpp 
    ${_binary_headers_root}/deprecated.hpp
    ${_binary_headers_root}/warning.hpp)

# --- Template instantiation ---
set(${PROJECT_NAME}_TEMPLATE_INSTANTIATION_PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/include/simplex/core/constraints-problem.txx
    ${PROJECT_SOURCE_DIR}/include/simplex/core/simulator.txx)

set(${PROJECT_NAME}_TEMPLATE_INSTANTIATION_SOURCES core/constraints-problem.cpp core/simulator.cpp)

# --- Pinocchio template instantiation ---
set(${PROJECT_NAME}_PINOCCHIO_TEMPLATE_INSTANTIATION_SOURCES
    pinocchio_template_instantiation/aba-derivatives.cpp 
    pinocchio_template_instantiation/aba.cpp
    pinocchio_template_instantiation/crba.cpp 
    pinocchio_template_instantiation/joint-model.cpp)

set(${PROJECT_NAME}_PINOCCHIO_TEMPLATE_INSTANTIATION_HEADERS
    ${PROJECT_SOURCE_DIR}/include/simplex/pinocchio_template_instantiation/aba-derivatives.txx
    ${PROJECT_SOURCE_DIR}/include/simplex/pinocchio_template_instantiation/aba.txx
    ${PROJECT_SOURCE_DIR}/include/simplex/pinocchio_template_instantiation/crba.txx
    ${PROJECT_SOURCE_DIR}/include/simplex/pinocchio_template_instantiation/joint-model.txx)

# --- Python bindings ---
set(${PROJECT_NAME}_BINDINGS_PYTHON_PUBLIC_HEADERS
    ${PROJECT_SOURCE_DIR}/include/simplex/bindings/python/fwd.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/bindings/python/core/constraints-problem.hpp
    ${PROJECT_SOURCE_DIR}/include/simplex/bindings/python/core/simulator.hpp)

set(${PROJECT_NAME}_BINDINGS_PYTHON_SOURCES
    ${PROJECT_SOURCE_DIR}/bindings/python/core/expose-contact-frame.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/core/expose-constraints-problem.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/core/expose-simulator.cpp
    ${PROJECT_SOURCE_DIR}/bindings/python/module.cpp)