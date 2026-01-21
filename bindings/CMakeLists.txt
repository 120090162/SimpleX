#
# Copyright (c) 2024 INRIA
#

add_subdirectory(python)
if(BUILD_PYTHON_INTERFACE)
	set(PYWRAP
		${PYWRAP}
		PARENT_SCOPE)
endif(BUILD_PYTHON_INTERFACE)
