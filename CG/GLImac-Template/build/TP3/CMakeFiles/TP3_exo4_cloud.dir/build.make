# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build

# Include any dependencies generated for this target.
include TP3/CMakeFiles/TP3_exo4_cloud.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include TP3/CMakeFiles/TP3_exo4_cloud.dir/compiler_depend.make

# Include the progress variables for this target.
include TP3/CMakeFiles/TP3_exo4_cloud.dir/progress.make

# Include the compile flags for this target's objects.
include TP3/CMakeFiles/TP3_exo4_cloud.dir/flags.make

TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o: TP3/CMakeFiles/TP3_exo4_cloud.dir/flags.make
TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o: ../TP3/exo4_cloud.cpp
TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o: TP3/CMakeFiles/TP3_exo4_cloud.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o"
	cd /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o -MF CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o.d -o CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o -c /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/TP3/exo4_cloud.cpp

TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.i"
	cd /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/TP3/exo4_cloud.cpp > CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.i

TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.s"
	cd /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/TP3/exo4_cloud.cpp -o CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.s

# Object files for target TP3_exo4_cloud
TP3_exo4_cloud_OBJECTS = \
"CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o"

# External object files for target TP3_exo4_cloud
TP3_exo4_cloud_EXTERNAL_OBJECTS =

TP3/TP3_exo4_cloud: TP3/CMakeFiles/TP3_exo4_cloud.dir/exo4_cloud.cpp.o
TP3/TP3_exo4_cloud: TP3/CMakeFiles/TP3_exo4_cloud.dir/build.make
TP3/TP3_exo4_cloud: glimac/libglimac.a
TP3/TP3_exo4_cloud: /usr/lib/x86_64-linux-gnu/libSDLmain.a
TP3/TP3_exo4_cloud: /usr/lib/x86_64-linux-gnu/libSDL.so
TP3/TP3_exo4_cloud: /usr/lib/x86_64-linux-gnu/libGL.so.1
TP3/TP3_exo4_cloud: ../third-party/libGLEW.a
TP3/TP3_exo4_cloud: TP3/CMakeFiles/TP3_exo4_cloud.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TP3_exo4_cloud"
	cd /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TP3_exo4_cloud.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
TP3/CMakeFiles/TP3_exo4_cloud.dir/build: TP3/TP3_exo4_cloud
.PHONY : TP3/CMakeFiles/TP3_exo4_cloud.dir/build

TP3/CMakeFiles/TP3_exo4_cloud.dir/clean:
	cd /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3 && $(CMAKE_COMMAND) -P CMakeFiles/TP3_exo4_cloud.dir/cmake_clean.cmake
.PHONY : TP3/CMakeFiles/TP3_exo4_cloud.dir/clean

TP3/CMakeFiles/TP3_exo4_cloud.dir/depend:
	cd /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/TP3 /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3 /home/stephanev/Documents/M2_2023_2024/CG/GLImac-Template/build/TP3/CMakeFiles/TP3_exo4_cloud.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : TP3/CMakeFiles/TP3_exo4_cloud.dir/depend

