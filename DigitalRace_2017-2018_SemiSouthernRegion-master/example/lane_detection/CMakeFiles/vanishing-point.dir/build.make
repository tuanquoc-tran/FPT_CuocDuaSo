# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example

# Include any dependencies generated for this target.
include lane_detection/CMakeFiles/vanishing-point.dir/depend.make

# Include the progress variables for this target.
include lane_detection/CMakeFiles/vanishing-point.dir/progress.make

# Include the compile flags for this target's objects.
include lane_detection/CMakeFiles/vanishing-point.dir/flags.make

lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o: lane_detection/CMakeFiles/vanishing-point.dir/flags.make
lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o: lane_detection/msac/MSAC.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o -c /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/MSAC.cpp

lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.i"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/MSAC.cpp > CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.i

lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.s"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/MSAC.cpp -o CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.s

lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.requires:
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.requires

lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.provides: lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.requires
	$(MAKE) -f lane_detection/CMakeFiles/vanishing-point.dir/build.make lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.provides.build
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.provides

lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.provides.build: lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o

lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o: lane_detection/CMakeFiles/vanishing-point.dir/flags.make
lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o: lane_detection/msac/errorNIETO.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o -c /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/errorNIETO.cpp

lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.i"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/errorNIETO.cpp > CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.i

lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.s"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/errorNIETO.cpp -o CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.s

lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.requires:
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.requires

lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.provides: lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.requires
	$(MAKE) -f lane_detection/CMakeFiles/vanishing-point.dir/build.make lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.provides.build
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.provides

lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.provides.build: lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o

lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o: lane_detection/CMakeFiles/vanishing-point.dir/flags.make
lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o: lane_detection/msac/lmmin.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o -c /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/lmmin.cpp

lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.i"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/lmmin.cpp > CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.i

lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.s"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/msac/lmmin.cpp -o CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.s

lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.requires:
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.requires

lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.provides: lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.requires
	$(MAKE) -f lane_detection/CMakeFiles/vanishing-point.dir/build.make lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.provides.build
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.provides

lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.provides.build: lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o

lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o: lane_detection/CMakeFiles/vanishing-point.dir/flags.make
lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o: lane_detection/api_lane_detection.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o -c /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/api_lane_detection.cpp

lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.i"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/api_lane_detection.cpp > CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.i

lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.s"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/api_lane_detection.cpp -o CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.s

lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.requires:
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.requires

lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.provides: lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.requires
	$(MAKE) -f lane_detection/CMakeFiles/vanishing-point.dir/build.make lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.provides.build
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.provides

lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.provides.build: lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o

# Object files for target vanishing-point
vanishing__point_OBJECTS = \
"CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o" \
"CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o" \
"CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o" \
"CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o"

# External object files for target vanishing-point
vanishing__point_EXTERNAL_OBJECTS =

bin/Release/libvanishing-point.a: lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o
bin/Release/libvanishing-point.a: lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o
bin/Release/libvanishing-point.a: lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o
bin/Release/libvanishing-point.a: lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o
bin/Release/libvanishing-point.a: lane_detection/CMakeFiles/vanishing-point.dir/build.make
bin/Release/libvanishing-point.a: lane_detection/CMakeFiles/vanishing-point.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library ../bin/Release/libvanishing-point.a"
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && $(CMAKE_COMMAND) -P CMakeFiles/vanishing-point.dir/cmake_clean_target.cmake
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vanishing-point.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lane_detection/CMakeFiles/vanishing-point.dir/build: bin/Release/libvanishing-point.a
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/build

lane_detection/CMakeFiles/vanishing-point.dir/requires: lane_detection/CMakeFiles/vanishing-point.dir/msac/MSAC.cpp.o.requires
lane_detection/CMakeFiles/vanishing-point.dir/requires: lane_detection/CMakeFiles/vanishing-point.dir/msac/errorNIETO.cpp.o.requires
lane_detection/CMakeFiles/vanishing-point.dir/requires: lane_detection/CMakeFiles/vanishing-point.dir/msac/lmmin.cpp.o.requires
lane_detection/CMakeFiles/vanishing-point.dir/requires: lane_detection/CMakeFiles/vanishing-point.dir/api_lane_detection.cpp.o.requires
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/requires

lane_detection/CMakeFiles/vanishing-point.dir/clean:
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection && $(CMAKE_COMMAND) -P CMakeFiles/vanishing-point.dir/cmake_clean.cmake
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/clean

lane_detection/CMakeFiles/vanishing-point.dir/depend:
	cd /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection /home/ubuntu/DriverlessCarChallenge_2017-2018-master/example/lane_detection/CMakeFiles/vanishing-point.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lane_detection/CMakeFiles/vanishing-point.dir/depend

