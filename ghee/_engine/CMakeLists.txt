cmake_minimum_required(VERSION 3.16)
project(ghee C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -funroll-loops")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")

find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM   REQUIRED libdrm)
pkg_check_modules(GBM   REQUIRED gbm)
pkg_check_modules(EGL   REQUIRED egl)
pkg_check_modules(GLES  REQUIRED glesv2)
pkg_check_modules(JPEG  REQUIRED libjpeg)

add_library(ghee SHARED engine.c io.c)
set_target_properties(ghee PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ghee PRIVATE
    ${DRM_INCLUDE_DIRS} ${GBM_INCLUDE_DIRS} ${EGL_INCLUDE_DIRS}
    ${GLES_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS})
target_link_libraries(ghee PRIVATE
    ${DRM_LIBRARIES} ${GBM_LIBRARIES} ${EGL_LIBRARIES} ${GLES_LIBRARIES} ${JPEG_LIBRARIES} m)
install(TARGETS ghee LIBRARY DESTINATION lib)
