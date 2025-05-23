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
CMAKE_SOURCE_DIR = /home/cosmos/alphazero-multi-game

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cosmos/alphazero-multi-game/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/core_tests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/core_tests.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/core_tests.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/core_tests.dir/flags.make

tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o: tests/CMakeFiles/core_tests.dir/flags.make
tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o: ../tests/core/game_factory_test.cpp
tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o: tests/CMakeFiles/core_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cosmos/alphazero-multi-game/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o -MF CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o.d -o CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o -c /home/cosmos/alphazero-multi-game/tests/core/game_factory_test.cpp

tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core_tests.dir/core/game_factory_test.cpp.i"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cosmos/alphazero-multi-game/tests/core/game_factory_test.cpp > CMakeFiles/core_tests.dir/core/game_factory_test.cpp.i

tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core_tests.dir/core/game_factory_test.cpp.s"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cosmos/alphazero-multi-game/tests/core/game_factory_test.cpp -o CMakeFiles/core_tests.dir/core/game_factory_test.cpp.s

tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o: tests/CMakeFiles/core_tests.dir/flags.make
tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o: ../tests/core/igamestate_test.cpp
tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o: tests/CMakeFiles/core_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cosmos/alphazero-multi-game/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o -MF CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o.d -o CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o -c /home/cosmos/alphazero-multi-game/tests/core/igamestate_test.cpp

tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core_tests.dir/core/igamestate_test.cpp.i"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cosmos/alphazero-multi-game/tests/core/igamestate_test.cpp > CMakeFiles/core_tests.dir/core/igamestate_test.cpp.i

tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core_tests.dir/core/igamestate_test.cpp.s"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cosmos/alphazero-multi-game/tests/core/igamestate_test.cpp -o CMakeFiles/core_tests.dir/core/igamestate_test.cpp.s

tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o: tests/CMakeFiles/core_tests.dir/flags.make
tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o: ../tests/core/zobrist_test.cpp
tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o: tests/CMakeFiles/core_tests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cosmos/alphazero-multi-game/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o -MF CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o.d -o CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o -c /home/cosmos/alphazero-multi-game/tests/core/zobrist_test.cpp

tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/core_tests.dir/core/zobrist_test.cpp.i"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cosmos/alphazero-multi-game/tests/core/zobrist_test.cpp > CMakeFiles/core_tests.dir/core/zobrist_test.cpp.i

tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/core_tests.dir/core/zobrist_test.cpp.s"
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cosmos/alphazero-multi-game/tests/core/zobrist_test.cpp -o CMakeFiles/core_tests.dir/core/zobrist_test.cpp.s

# Object files for target core_tests
core_tests_OBJECTS = \
"CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o" \
"CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o" \
"CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o"

# External object files for target core_tests
core_tests_EXTERNAL_OBJECTS =

tests/core_tests: tests/CMakeFiles/core_tests.dir/core/game_factory_test.cpp.o
tests/core_tests: tests/CMakeFiles/core_tests.dir/core/igamestate_test.cpp.o
tests/core_tests: tests/CMakeFiles/core_tests.dir/core/zobrist_test.cpp.o
tests/core_tests: tests/CMakeFiles/core_tests.dir/build.make
tests/core_tests: src/libalphazero_lib.a
tests/core_tests: /usr/lib/x86_64-linux-gnu/libgtest.a
tests/core_tests: /usr/lib/x86_64-linux-gnu/libgtest_main.a
tests/core_tests: /usr/lib/x86_64-linux-gnu/libspdlog.so.1.9.2
tests/core_tests: /usr/lib/x86_64-linux-gnu/libfmt.so.8.1.1
tests/core_tests: src/nn/libalphazero_nn.a
tests/core_tests: src/mcts/libalphazero_mcts.a
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libc10.so
tests/core_tests: /usr/local/cuda-12.4/lib64/libnvrtc.so
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libc10_cuda.so
tests/core_tests: src/core/libalphazero_core.a
tests/core_tests: src/games/chess/libalphazero_chess.a
tests/core_tests: src/games/go/libalphazero_go.a
tests/core_tests: src/games/gomoku/libalphazero_gomoku.a
tests/core_tests: src/core/libalphazero_core.a
tests/core_tests: src/games/chess/libalphazero_chess.a
tests/core_tests: src/games/go/libalphazero_go.a
tests/core_tests: src/games/gomoku/libalphazero_gomoku.a
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libtorch.so
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libprotobuf.so.28.2.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_check_op.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_leak_check.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_die_if_null.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_conditions.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_message.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_nullguard.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_examine_stack.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_format.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_proto.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_log_sink_set.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_sink.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_entry.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_marshalling.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_reflection.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_config.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_program_name.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_private_handle_accessor.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_commandlineflag.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_flags_commandlineflag_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_initialize.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_globals.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_globals.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_vlog_config_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_internal_fnmatch.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_raw_hash_set.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_hash.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_city.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_low_level_hash.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_hashtablez_sampler.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_distributions.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_seed_sequences.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_pool_urbg.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_randen.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_randen_hwaes.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_randen_hwaes_impl.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_randen_slow.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_platform.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_internal_seed_material.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_random_seed_gen_exception.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_statusor.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_status.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_cord.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_cordz_info.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_cord_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_cordz_functions.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_exponential_biased.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_cordz_handle.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_crc_cord_state.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_crc32c.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_crc_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_crc_cpu_detect.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_bad_optional_access.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_strerror.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_str_format_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_synchronization.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_stacktrace.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_symbolize.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_debugging_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_demangle_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_demangle_rust.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_decode_rust_punycode.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_utf8_for_code_point.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_graphcycles_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_kernel_timeout_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_malloc_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_time.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_civil_time.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_time_zone.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_bad_variant_access.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_strings.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_int128.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_strings_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_string_view.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_base.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_spinlock_wait.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_throw_delegate.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_raw_logging_internal.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libabsl_log_severity.so.2407.0.0
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libc10_cuda.so
tests/core_tests: /home/cosmos/anaconda3/envs/alphazero_env/lib/libc10.so
tests/core_tests: /usr/local/cuda-12.4/lib64/libcudart.so
tests/core_tests: /usr/lib/x86_64-linux-gnu/libcudnn.so
tests/core_tests: /usr/lib/x86_64-linux-gnu/libgtest.a
tests/core_tests: tests/CMakeFiles/core_tests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cosmos/alphazero-multi-game/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable core_tests"
	cd /home/cosmos/alphazero-multi-game/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/core_tests.dir/link.txt --verbose=$(VERBOSE)
	cd /home/cosmos/alphazero-multi-game/build/tests && /usr/bin/cmake -D TEST_TARGET=core_tests -D TEST_EXECUTABLE=/home/cosmos/alphazero-multi-game/build/tests/core_tests -D TEST_EXECUTOR= -D TEST_WORKING_DIR=/home/cosmos/alphazero-multi-game/build/tests -D TEST_EXTRA_ARGS= -D TEST_PROPERTIES= -D TEST_PREFIX= -D TEST_SUFFIX= -D TEST_FILTER= -D NO_PRETTY_TYPES=FALSE -D NO_PRETTY_VALUES=FALSE -D TEST_LIST=core_tests_TESTS -D CTEST_FILE=/home/cosmos/alphazero-multi-game/build/tests/core_tests[1]_tests.cmake -D TEST_DISCOVERY_TIMEOUT=5 -D TEST_XML_OUTPUT_DIR= -P /usr/share/cmake-3.22/Modules/GoogleTestAddTests.cmake

# Rule to build all files generated by this target.
tests/CMakeFiles/core_tests.dir/build: tests/core_tests
.PHONY : tests/CMakeFiles/core_tests.dir/build

tests/CMakeFiles/core_tests.dir/clean:
	cd /home/cosmos/alphazero-multi-game/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/core_tests.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/core_tests.dir/clean

tests/CMakeFiles/core_tests.dir/depend:
	cd /home/cosmos/alphazero-multi-game/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cosmos/alphazero-multi-game /home/cosmos/alphazero-multi-game/tests /home/cosmos/alphazero-multi-game/build /home/cosmos/alphazero-multi-game/build/tests /home/cosmos/alphazero-multi-game/build/tests/CMakeFiles/core_tests.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/core_tests.dir/depend

