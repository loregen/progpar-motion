# 3.11 because of imported target "OpenMP::OpenMP_C"
cmake_minimum_required(VERSION 3.11)

project(motion)

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# CMake options ---------------------------------------------------------------
# -----------------------------------------------------------------------------
option(MOTION_EXE "compile the detection chain executable." ON)
option(MOTION_OPENMP_LINK "link with OpenMP library." ON)
option(MOTION_OPENCV_LINK "link with OpenCV library." OFF)
option(MOTION_SPU_RUNTIME "compile and link with StreamPU runtime." OFF)

if (MOTION_OPENCV_LINK OR MOTION_SPU_RUNTIME)
	set(MOTION_CPP ON)
else()
	set(MOTION_CPP OFF)
endif()

# Print CMake options values --------------------------------------------------
# -----------------------------------------------------------------------------
message(STATUS "Motion options: ")
message(STATUS "  * MOTION_EXE: '${MOTION_EXE}'")
message(STATUS "  * MOTION_OPENMP_LINK: '${MOTION_OPENMP_LINK}'")
message(STATUS "  * MOTION_OPENCV_LINK: '${MOTION_OPENCV_LINK}'")
message(STATUS "  * MOTION_SPU_RUNTIME: '${MOTION_SPU_RUNTIME}'")
message(STATUS "Motion info: ")
message(STATUS "  * MOTION_CPP: '${MOTION_CPP}'")
message(STATUS "  * CMAKE_BUILD_TYPE: '${CMAKE_BUILD_TYPE}'")

# Check CMake options ---------------------------------------------------------
# -----------------------------------------------------------------------------
# empty...

# Define CMake variables ------------------------------------------------------
# -----------------------------------------------------------------------------
set(src_dir src)
set(inc_dir include)
set(exe_dir bin)
set(lib_dir lib)

# Compiler generic options ----------------------------------------------------
# -----------------------------------------------------------------------------
# enable C11
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if (MOTION_CPP)
	# enable C++11
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

# Specify the executable and lib output path ----------------------------------
# -----------------------------------------------------------------------------
set(EXECUTABLE_OUTPUT_PATH ${exe_dir})

# Check c-vector mandatory dependency -----------------------------------------
# -----------------------------------------------------------------------------
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/c-vector/vec.h")
	message(FATAL_ERROR "'c-vector' can't be found, try to init the submodule with the following cmd:\n"
	                    "$ git submodule update --init -- ../lib/c-vector/")
endif()

# Declare source files to compile ---------------------------------------------
# -----------------------------------------------------------------------------
set(src_common_files
    ${src_dir}/common/args.c
    ${src_dir}/common/tools.c
    ${src_dir}/common/CCL/CCL_compute.c
    ${src_dir}/common/features/features_compute.c
    ${src_dir}/common/features/features_io.c
    ${src_dir}/common/image/image_compute.c
    ${src_dir}/common/image/image_struct.c
    ${src_dir}/common/kNN/kNN_compute.c
    ${src_dir}/common/kNN/kNN_io.c
    ${src_dir}/common/morpho/morpho_compute.c
    ${src_dir}/common/sigma_delta/sigma_delta_compute.c
    ${src_dir}/common/tracking/tracking_compute.c
    ${src_dir}/common/tracking/tracking_io.c
    ${src_dir}/common/tracking/tracking_struct.c
    ${src_dir}/common/video/video_io.c
    ${src_dir}/common/video/video_struct.c
    ${src_dir}/common/visu/visu_io.c
    ${lib_dir}/c-vector/vec.c)
list(APPEND motion_src_list ${src_common_files})

if (MOTION_CPP)
	file(GLOB_RECURSE src_common_cpp_files ${src_dir}/common/tools.cpp
	                                       ${src_dir}/common/args.cpp)
    if (MOTION_SPU_RUNTIME)
        file(GLOB_RECURSE src_common_cpp_files ${src_dir}/wrapper/*)
    endif()
endif()

# Create binaries -------------------------------------------------------------
# -----------------------------------------------------------------------------
# objects
add_library(motion-common-obj OBJECT ${src_common_files})
list(APPEND motion_targets_list motion-common-obj)
if (MOTION_CPP)
	add_library(motion-common-cpp-obj OBJECT ${src_common_cpp_files})
	list(APPEND motion_targets_list motion-common-cpp-obj)
endif()

# executables
if(MOTION_EXE)
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_dir}/main/motion.c")
		set(src_motion_files ${src_dir}/main/motion.c)
		list(APPEND motion_src_list ${src_motion_files})
		if (MOTION_CPP)
			add_executable(motion-exe $<TARGET_OBJECTS:motion-common-obj> $<TARGET_OBJECTS:motion-common-cpp-obj> ${src_motion_files})
		else()
			add_executable(motion-exe $<TARGET_OBJECTS:motion-common-obj> ${src_motion_files})
		endif()
		list(APPEND motion_targets_list motion-exe)
		set_target_properties(motion-exe PROPERTIES OUTPUT_NAME motion)
	endif()

	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_dir}/main/motion2.c")
		set(src_motion2_files ${src_dir}/main/motion2.c)
		list(APPEND motion_src_list ${src_motion2_files})
		if (MOTION_CPP)
			add_executable(motion2-exe $<TARGET_OBJECTS:motion-common-obj> $<TARGET_OBJECTS:motion-common-cpp-obj> ${src_motion2_files})
		else()
			add_executable(motion2-exe $<TARGET_OBJECTS:motion-common-obj> ${src_motion2_files})
		endif()
		list(APPEND motion_targets_list motion2-exe)
		set_target_properties(motion2-exe PROPERTIES OUTPUT_NAME motion2)
	endif()

	if (MOTION_SPU_RUNTIME)
		if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_dir}/main/motion_spu.cpp")
			set(src_motion_spu_files ${src_dir}/main/motion_spu.cpp)
			list(APPEND motion_src_list ${src_motion_spu_files})
			add_executable(motion-spu-exe $<TARGET_OBJECTS:motion-common-obj> $<TARGET_OBJECTS:motion-common-cpp-obj> ${src_motion_spu_files})
			list(APPEND motion_targets_list motion-spu-exe)
			set_target_properties(motion-spu-exe PROPERTIES OUTPUT_NAME motion-spu)
		endif()

		if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_dir}/main/motion2_spu.cpp")
			set(src_motion2_spu_files ${src_dir}/main/motion2_spu.cpp)
			list(APPEND motion_src_list ${src_motion2_spu_files})
			add_executable(motion2-spu-exe $<TARGET_OBJECTS:motion-common-obj> $<TARGET_OBJECTS:motion-common-cpp-obj> ${src_motion2_spu_files})
			list(APPEND motion_targets_list motion2-spu-exe)
			set_target_properties(motion2-spu-exe PROPERTIES OUTPUT_NAME motion2-spu)
		endif()
	endif()
endif()

macro(motion_set_source_files_properties files key value)
	foreach(_file IN ITEMS ${files})
		set_source_files_properties(${_file} PROPERTIES ${key} ${value})
	endforeach()
endmacro()

# force C++ language when linking with OpenCV
if (MOTION_CPP)
	motion_set_source_files_properties("${motion_src_list}" LANGUAGE CXX)
endif()

# Add and compile sub-projects ------------------------------------------------
# -----------------------------------------------------------------------------
# ffmpeg-io
if (NOT TARGET ffmpeg-io-slib)
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/ffmpeg-io/include/constants.h")
		option(FFMPEG-IO_EXE "Forced option from '${CMAKE_PROJECT_NAME}'." OFF)
		add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/ffmpeg-io/)
	else()
		message(FATAL_ERROR "'ffmpeg-io' can't be found, try to init the submodule with the following cmd:\n"
		                    "$ git submodule update --init -- ../lib/ffmpeg-io/")
	endif()
endif()

# nrc
if (NOT TARGET nrc-slib)
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/nrc2/include/nrc.h")
		add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/nrc2/)
	else()
		message(FATAL_ERROR "'nrc2' can't be found, try to init the submodule with the following cmd:\n"
		                    "$ git submodule update --init -- ../lib/nrc2/")
	endif()
endif()

# OpenCV
if (MOTION_OPENCV_LINK)
	find_package(OpenCV REQUIRED)
endif()

# StreamPU
if (MOTION_SPU_RUNTIME)
	if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/streampu/include/streampu.hpp")
		option(SPU_COMPILE_STATIC_LIB "" ON)
		option(SPU_COMPILE_SHARED_LIB "" OFF)
		option(SPU_TESTS "" OFF)
		add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/streampu/)
	else()
		message(FATAL_ERROR "'StreamPU' can't be found, try to init the submodule with the following cmd:\n"
		                    "$ git submodule update --init --recursive -- ../lib/streampu/")
	endif()
endif()

# Add definitions -------------------------------------------------------------
# -----------------------------------------------------------------------------
macro(motion_target_compile_definitions targets privacy dir)
	foreach(_target IN ITEMS ${targets})
		target_compile_definitions(${_target} ${privacy} ${dir})
	endforeach()
endmacro()

if (MOTION_OPENCV_LINK)
	motion_target_compile_definitions("${motion_targets_list}" PUBLIC MOTION_OPENCV_LINK)
endif()
if (MOTION_SPU_RUNTIME)
	motion_target_compile_definitions("${motion_targets_list}" PUBLIC MOTION_SPU_LINK)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mipp/include/mipp.h")
	motion_target_compile_definitions("${motion_targets_list}" PUBLIC MIPP_ENABLE_BACKTRACE)
endif()


# Keep ffmpeg-io enabled
motion_target_compile_definitions("${motion_targets_list}" PUBLIC MOTION_USE_FFMPEG_IO)

# Set include directory -------------------------------------------------------
# -----------------------------------------------------------------------------
macro(motion_target_include_directories targets privacy dir)
	foreach(_target IN ITEMS ${targets})
		target_include_directories(${_target} ${privacy} ${dir})
	endforeach()
endmacro()

motion_target_include_directories("${motion_targets_list}" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${inc_dir}/c)
if (MOTION_CPP)
	motion_target_include_directories("${motion_targets_list}" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${inc_dir}/cpp)
endif()
motion_target_include_directories("${motion_targets_list}" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${lib_dir}/c-vector)
if (MOTION_OPENCV_LINK)
	motion_target_include_directories("${motion_targets_list}" PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()
if (MOTION_SPU_RUNTIME)
	motion_target_include_directories("${motion_targets_list}" PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/${inc_dir}/cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/mipp/include/mipp.h")
	motion_target_include_directories("${motion_targets_list}" PUBLIC "lib/mipp/include")
	message(STATUS "Motion - Header found: MIPP")
endif()

# Link with dependencies ------------------------------------------------------
# -----------------------------------------------------------------------------
macro(motion_target_link_libraries targets privacy lib)
	foreach(_target IN ITEMS ${targets})
		target_link_libraries(${_target} ${privacy} ${lib})
	endforeach()
endmacro()

motion_target_link_libraries("${motion_targets_list}" PUBLIC ffmpeg-io-slib)
motion_target_link_libraries("${motion_targets_list}" PUBLIC m)
motion_target_link_libraries("${motion_targets_list}" PUBLIC nrc-slib)
if(MOTION_OPENMP_LINK)
	find_package(OpenMP REQUIRED)
	if (MOTION_CPP)
		motion_target_link_libraries("${motion_targets_list}" PRIVATE OpenMP::OpenMP_CXX)
	else()
		motion_target_link_libraries("${motion_targets_list}" PRIVATE OpenMP::OpenMP_C)
	endif()
endif()
if (MOTION_OPENCV_LINK)
	motion_target_link_libraries("${motion_targets_list}" PUBLIC "${OpenCV_LIBS}")
endif()
if (MOTION_SPU_RUNTIME)
	motion_target_link_libraries("${motion_targets_list}" PUBLIC spu-static-lib)
endif()
