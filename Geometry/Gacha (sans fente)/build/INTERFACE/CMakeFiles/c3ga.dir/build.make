# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.9/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.9/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build"

# Include any dependencies generated for this target.
include INTERFACE/CMakeFiles/c3ga.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include INTERFACE/CMakeFiles/c3ga.dir/compiler_depend.make

# Include the progress variables for this target.
include INTERFACE/CMakeFiles/c3ga.dir/progress.make

# Include the compile flags for this target's objects.
include INTERFACE/CMakeFiles/c3ga.dir/flags.make

INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o: INTERFACE/CMakeFiles/c3ga.dir/flags.make
INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o: /home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha\ (sans\ fente)/lib/garamon_c3ga/src/c3ga/Mvec.cpp
INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o: INTERFACE/CMakeFiles/c3ga.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o"
	cd "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o -MF CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o.d -o CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o -c "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/lib/garamon_c3ga/src/c3ga/Mvec.cpp"

INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.i"
	cd "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/lib/garamon_c3ga/src/c3ga/Mvec.cpp" > CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.i

INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.s"
	cd "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE" && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/lib/garamon_c3ga/src/c3ga/Mvec.cpp" -o CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.s

# Object files for target c3ga
c3ga_OBJECTS = \
"CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o"

# External object files for target c3ga
c3ga_EXTERNAL_OBJECTS =

INTERFACE/libc3ga.so: INTERFACE/CMakeFiles/c3ga.dir/src/c3ga/Mvec.cpp.o
INTERFACE/libc3ga.so: INTERFACE/CMakeFiles/c3ga.dir/build.make
INTERFACE/libc3ga.so: INTERFACE/CMakeFiles/c3ga.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libc3ga.so"
	cd "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/c3ga.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
INTERFACE/CMakeFiles/c3ga.dir/build: INTERFACE/libc3ga.so
.PHONY : INTERFACE/CMakeFiles/c3ga.dir/build

INTERFACE/CMakeFiles/c3ga.dir/clean:
	cd "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE" && $(CMAKE_COMMAND) -P CMakeFiles/c3ga.dir/cmake_clean.cmake
.PHONY : INTERFACE/CMakeFiles/c3ga.dir/clean

INTERFACE/CMakeFiles/c3ga.dir/depend:
	cd "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)" "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/lib/garamon_c3ga" "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build" "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE" "/home/2ing2/stephane.vong/Documents/M2_2023_2024/Geometry/Gacha (sans fente)/build/INTERFACE/CMakeFiles/c3ga.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : INTERFACE/CMakeFiles/c3ga.dir/depend

