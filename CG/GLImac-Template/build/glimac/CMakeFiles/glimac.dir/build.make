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
include glimac/CMakeFiles/glimac.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include glimac/CMakeFiles/glimac.dir/compiler_depend.make

# Include the progress variables for this target.
include glimac/CMakeFiles/glimac.dir/progress.make

# Include the compile flags for this target's objects.
include glimac/CMakeFiles/glimac.dir/flags.make

glimac/CMakeFiles/glimac.dir/src/Cone.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/Cone.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Cone.cpp
glimac/CMakeFiles/glimac.dir/src/Cone.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object glimac/CMakeFiles/glimac.dir/src/Cone.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/Cone.cpp.o -MF CMakeFiles/glimac.dir/src/Cone.cpp.o.d -o CMakeFiles/glimac.dir/src/Cone.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Cone.cpp

glimac/CMakeFiles/glimac.dir/src/Cone.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/Cone.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Cone.cpp > CMakeFiles/glimac.dir/src/Cone.cpp.i

glimac/CMakeFiles/glimac.dir/src/Cone.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/Cone.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Cone.cpp -o CMakeFiles/glimac.dir/src/Cone.cpp.s

glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Geometry.cpp
glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.o -MF CMakeFiles/glimac.dir/src/Geometry.cpp.o.d -o CMakeFiles/glimac.dir/src/Geometry.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Geometry.cpp

glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/Geometry.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Geometry.cpp > CMakeFiles/glimac.dir/src/Geometry.cpp.i

glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/Geometry.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Geometry.cpp -o CMakeFiles/glimac.dir/src/Geometry.cpp.s

glimac/CMakeFiles/glimac.dir/src/Image.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/Image.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Image.cpp
glimac/CMakeFiles/glimac.dir/src/Image.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object glimac/CMakeFiles/glimac.dir/src/Image.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/Image.cpp.o -MF CMakeFiles/glimac.dir/src/Image.cpp.o.d -o CMakeFiles/glimac.dir/src/Image.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Image.cpp

glimac/CMakeFiles/glimac.dir/src/Image.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/Image.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Image.cpp > CMakeFiles/glimac.dir/src/Image.cpp.i

glimac/CMakeFiles/glimac.dir/src/Image.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/Image.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Image.cpp -o CMakeFiles/glimac.dir/src/Image.cpp.s

glimac/CMakeFiles/glimac.dir/src/Program.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/Program.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Program.cpp
glimac/CMakeFiles/glimac.dir/src/Program.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object glimac/CMakeFiles/glimac.dir/src/Program.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/Program.cpp.o -MF CMakeFiles/glimac.dir/src/Program.cpp.o.d -o CMakeFiles/glimac.dir/src/Program.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Program.cpp

glimac/CMakeFiles/glimac.dir/src/Program.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/Program.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Program.cpp > CMakeFiles/glimac.dir/src/Program.cpp.i

glimac/CMakeFiles/glimac.dir/src/Program.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/Program.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Program.cpp -o CMakeFiles/glimac.dir/src/Program.cpp.s

glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/SDLWindowManager.cpp
glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o -MF CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o.d -o CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/SDLWindowManager.cpp

glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/SDLWindowManager.cpp > CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.i

glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/SDLWindowManager.cpp -o CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.s

glimac/CMakeFiles/glimac.dir/src/Shader.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/Shader.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Shader.cpp
glimac/CMakeFiles/glimac.dir/src/Shader.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object glimac/CMakeFiles/glimac.dir/src/Shader.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/Shader.cpp.o -MF CMakeFiles/glimac.dir/src/Shader.cpp.o.d -o CMakeFiles/glimac.dir/src/Shader.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Shader.cpp

glimac/CMakeFiles/glimac.dir/src/Shader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/Shader.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Shader.cpp > CMakeFiles/glimac.dir/src/Shader.cpp.i

glimac/CMakeFiles/glimac.dir/src/Shader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/Shader.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Shader.cpp -o CMakeFiles/glimac.dir/src/Shader.cpp.s

glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Sphere.cpp
glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.o -MF CMakeFiles/glimac.dir/src/Sphere.cpp.o.d -o CMakeFiles/glimac.dir/src/Sphere.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Sphere.cpp

glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/Sphere.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Sphere.cpp > CMakeFiles/glimac.dir/src/Sphere.cpp.i

glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/Sphere.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/Sphere.cpp -o CMakeFiles/glimac.dir/src/Sphere.cpp.s

glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o: glimac/CMakeFiles/glimac.dir/flags.make
glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o: /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/tiny_obj_loader.cpp
glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o: glimac/CMakeFiles/glimac.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o -MF CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o.d -o CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o -c /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/tiny_obj_loader.cpp

glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.i"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/tiny_obj_loader.cpp > CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.i

glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.s"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac/src/tiny_obj_loader.cpp -o CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.s

# Object files for target glimac
glimac_OBJECTS = \
"CMakeFiles/glimac.dir/src/Cone.cpp.o" \
"CMakeFiles/glimac.dir/src/Geometry.cpp.o" \
"CMakeFiles/glimac.dir/src/Image.cpp.o" \
"CMakeFiles/glimac.dir/src/Program.cpp.o" \
"CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o" \
"CMakeFiles/glimac.dir/src/Shader.cpp.o" \
"CMakeFiles/glimac.dir/src/Sphere.cpp.o" \
"CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o"

# External object files for target glimac
glimac_EXTERNAL_OBJECTS =

glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/Cone.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/Geometry.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/Image.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/Program.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/SDLWindowManager.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/Shader.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/Sphere.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/src/tiny_obj_loader.cpp.o
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/build.make
glimac/libglimac.a: glimac/CMakeFiles/glimac.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX static library libglimac.a"
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && $(CMAKE_COMMAND) -P CMakeFiles/glimac.dir/cmake_clean_target.cmake
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/glimac.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
glimac/CMakeFiles/glimac.dir/build: glimac/libglimac.a
.PHONY : glimac/CMakeFiles/glimac.dir/build

glimac/CMakeFiles/glimac.dir/clean:
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac && $(CMAKE_COMMAND) -P CMakeFiles/glimac.dir/cmake_clean.cmake
.PHONY : glimac/CMakeFiles/glimac.dir/clean

glimac/CMakeFiles/glimac.dir/depend:
	cd /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/2ing2/stephane.vong/Documents/CG/GLImac-Template /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/glimac /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac /home/2ing2/stephane.vong/Documents/CG/GLImac-Template/build/glimac/CMakeFiles/glimac.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : glimac/CMakeFiles/glimac.dir/depend

