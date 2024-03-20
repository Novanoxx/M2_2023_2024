# Install script for directory: /home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/c3ga/BasisTransformations.hpp;/usr/local/include/c3ga/Constants.hpp;/usr/local/include/c3ga/DualCoefficients.hpp;/usr/local/include/c3ga/Geometric.hpp;/usr/local/include/c3ga/GeometricExplicit.hpp;/usr/local/include/c3ga/Inner.hpp;/usr/local/include/c3ga/InnerExplicit.hpp;/usr/local/include/c3ga/Mvec.hpp;/usr/local/include/c3ga/Outer.hpp;/usr/local/include/c3ga/OuterExplicit.hpp;/usr/local/include/c3ga/Utility.hpp;/usr/local/include/c3ga/Mvec.cpp")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/include/c3ga" TYPE FILE FILES
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/BasisTransformations.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Constants.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/DualCoefficients.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Geometric.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/GeometricExplicit.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Inner.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/InnerExplicit.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Mvec.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Outer.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/OuterExplicit.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Utility.hpp"
    "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/lib/garamon_c3ga/src/c3ga/Mvec.cpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/libc3ga.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/libc3ga.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/usr/local/lib/libc3ga.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/libc3ga.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/usr/local/lib" TYPE SHARED_LIBRARY FILES "/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/build/INTERFACE/libc3ga.so")
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/libc3ga.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/libc3ga.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/usr/local/lib/libc3ga.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/stephanev/Documents/M2_2023_2024/Geometry/tp_c3ga_c++_correction/build/INTERFACE/doc/cmake_install.cmake")

endif()

