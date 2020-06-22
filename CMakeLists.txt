cmake_minimum_required(VERSION 3.8)

project(MultiX LANGUAGES CXX)

# indicate if OPENMP should be enabled
option(USE_OPENMP "Use OPENMP" ON)

# indicate if the sample project should be included
option(CREATE_SAMPLE_PROJECT "Create the Sample Project" ON)

# ==============================================================================
# Check C++17 support
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==============================================================================
# Find OpenCV
# ==============================================================================
find_package(OpenCV 3.0 REQUIRED)

# ==============================================================================
# Find OpenMP
# ==============================================================================
find_package(OpenMP)
if (USE_OPENMP)
	if(NOT OPENMP_FOUND)
		message(FATAL_ERROR "OPENMP not found.")
	endif()
	add_definitions(-DUSE_OPENMP)
	set(TRGT_LNK_LBS_ADDITIONAL OpenMP::OpenMP_CXX)
endif (USE_OPENMP)

# ==============================================================================
# Includes
# ==============================================================================
include_directories (
	${PROJECT_SOURCE_DIR}/include
)

# ==============================================================================
# Structure: Multi-X Library
# ==============================================================================
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Set header files for the library
file(GLOB_RECURSE HDRS_MULTIX
	"include/*.h"
) 

# Set source files to be added to the library
file(GLOB_RECURSE SRCS_MULTIX 
	"include/*.cpp"
)

add_library(${PROJECT_NAME} STATIC
	${HDRS_MULTIX}
	${SRCS_MULTIX}
)

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries (${PROJECT_NAME}
	${OpenCV_LIBS})
	
# ==============================================================================
# Structure: Applications
# ==============================================================================
if (CREATE_SAMPLE_PROJECT)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

	add_executable(SampleProject
		src/main.cpp)
		
	target_link_libraries(SampleProject 
		${PROJECT_NAME}
		${OpenCV_LIBS}
	)

	add_dependencies(SampleProject ${PROJECT_NAME})
endif (CREATE_SAMPLE_PROJECT)
