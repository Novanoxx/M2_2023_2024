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
CMAKE_SOURCE_DIR = /home/2ing2/stephane.vong/Documents/CG/GLImac-Template

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build

# Include any dependencies generated for this target.
include TP_template/CMakeFiles/TP_template_SDLtemplate.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include TP_template/CMakeFiles/TP_template_SDLtemplate.dir/compiler_depend.make

# Include the progress variables for this target.
include TP_template/CMakeFiles/TP_template_SDLtemplate.dir/progress.make

# Include the compile flags for this target's objects.
include TP_template/CMakeFiles/TP_template_SDLtemplate.dir/flags.make

TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o: TP_template/CMakeFiles/TP_template_SDLtemplate.dir/flags.make
TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/TP_template/SDLtemplate.cpp
TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o: TP_template/CMakeFiles/TP_template_SDLtemplate.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o -MF CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o.d -o CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/TP_template/SDLtemplate.cpp

TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/TP_template/SDLtemplate.cpp > CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.i

TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/TP_template/SDLtemplate.cpp -o CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.s

# Object files for target TP_template_SDLtemplate
TP_template_SDLtemplate_OBJECTS = \
"CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o"

# External object files for target TP_template_SDLtemplate
TP_template_SDLtemplate_EXTERNAL_OBJECTS =

TP_template/TP_template_SDLtemplate: TP_template/CMakeFiles/TP_template_SDLtemplate.dir/SDLtemplate.cpp.o
TP_template/TP_template_SDLtemplate: TP_template/CMakeFiles/TP_template_SDLtemplate.dir/build.make
TP_template/TP_template_SDLtemplate: glimac/libglimac.a
TP_template/TP_template_SDLtemplate: /usr/lib/x86_64-linux-gnu/libSDLmain.a
TP_template/TP_template_SDLtemplate: /usr/lib/x86_64-linux-gnu/libSDL.so
TP_template/TP_template_SDLtemplate: /usr/lib/x86_64-linux-gnu/libGL.so.1
TP_template/TP_template_SDLtemplate: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/third-party/libGLEW.a
TP_template/TP_template_SDLtemplate: TP_template/CMakeFiles/TP_template_SDLtemplate.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable TP_template_SDLtemplate"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/TP_template_SDLtemplate.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
TP_template/CMakeFiles/TP_template_SDLtemplate.dir/build: TP_template/TP_template_SDLtemplate
.PHONY : TP_template/CMakeFiles/TP_template_SDLtemplate.dir/build

TP_template/CMakeFiles/TP_template_SDLtemplate.dir/clean:
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template && $(CMAKE_COMMAND) -P CMakeFiles/TP_template_SDLtemplate.dir/cmake_clean.cmake
.PHONY : TP_template/CMakeFiles/TP_template_SDLtemplate.dir/clean

TP_template/CMakeFiles/TP_template_SDLtemplate.dir/depend:
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/2ing2/stephane.vong/Documents/CG/GLImac-Template /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/TP_template /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/TP_template/CMakeFiles/TP_template_SDLtemplate.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : TP_template/CMakeFiles/TP_template_SDLtemplate.dir/depend

