# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # For xcframework support
  cmake_minimum_required(VERSION 3.28)
else()
  cmake_minimum_required(VERSION 3.26)
endif()
include(FetchContent)
include(CMakeDependentOption)
project(Generators LANGUAGES C CXX)

# All Options should be defined in cmake/options.cmake This must be included before any other cmake file is included
include(cmake/options.cmake)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11")
endif()

# Avoid warning of Calling FetchContent_Populate(Lib) is deprecated temporarily
# TODO: find a better way to handle the header-only 3rd party deps
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0169 OLD)
endif()

if(MSVC)
  # DLL initialization errors due to old conda msvcp140.dll dll are a result of the new MSVC compiler
  # See https://developercommunity.visualstudio.com/t/Access-violation-with-std::mutex::lock-a/10664660#T-N10668856
  # Remove this definition once the conda msvcp140.dll dll is updated.
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)

  # Add compiler options for security standards
  # Note: Must use separate generator expressions — do NOT use "C,CXX" in COMPILE_LANGUAGE.

  # Explicitly enable (/GS) stack protection to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/GS>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GS>")

  # Enable Control Flow Guard on compiler and linker to pass BinSkim checks
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/guard:cf>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/guard:cf>")
  add_link_options(/guard:cf /DYNAMICBASE)

  # Enable CET shadow stack unless building for ARM64 (not supported there)
  message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL "ARM64")
    message(STATUS "CET shadow stack enabled (/CETCOMPAT)")
    add_link_options(/CETCOMPAT)
  else()
    message(STATUS "CET shadow stack skipped for ARM64 build")
  endif()

  # Enable Spectre mitigations for C and C++ compilations only (avoid CUDA nvcc - not supported there)
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C>:/Qspectre>"
    "$<$<COMPILE_LANGUAGE:CXX>:/Qspectre>"
  )

  # Use updated value for __cplusplus macro (e.g. 201703L for C++17) instead of default 199711L
  # Required by some libraries/tools that check __cplusplus for feature support
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)

  add_compile_options(
    # Suppress warning C5038: data member will be initialized after another
    # Common when member initializer list is written out-of-order vs. declaration
    "$<$<COMPILE_LANGUAGE:C>:/w15038>"
    "$<$<COMPILE_LANGUAGE:CXX>:/w15038>"

    # Suppress warning C4100: unreferenced formal parameter
    # Often appears in template or virtual function overrides where a param is unused
    "$<$<COMPILE_LANGUAGE:C>:/wd4100>"
    "$<$<COMPILE_LANGUAGE:CXX>:/wd4100>"

    # Enable warning level 4 (more aggressive than default /W3)
    # Captures more potential bugs or code smells
    "$<$<COMPILE_LANGUAGE:C>:/W4>"
    "$<$<COMPILE_LANGUAGE:CXX>:/W4>"

    # Treat all warnings as errors (fail build on any warning)
    # Combine carefully with /W4 — suppress any known false positives
    "$<$<COMPILE_LANGUAGE:C>:/WX>"
    "$<$<COMPILE_LANGUAGE:CXX>:/WX>"
  )
endif()

include(cmake/ortlib.cmake)
include(cmake/external/onnxruntime_external_deps.cmake)
# All Global variables, including GLOB, for the top level CMakeLists.txt should be defined here
include(cmake/global_variables.cmake)
# Checking if CUDA is supported
include(cmake/check_cuda.cmake)
# Checking if ROCm is supported
include(cmake/check_rocm.cmake)
# Checking if DML is supported
include(cmake/check_dml.cmake)

include(cmake/cxx_standard.cmake)

add_compile_definitions(BUILDING_ORT_GENAI_C)

add_compile_definitions(USE_GUIDANCE=$<BOOL:${USE_GUIDANCE}>)

# Suggested by https://gitlab.kitware.com/cmake/cmake/-/issues/20132
# MacCatalyst is not well supported in CMake
# The error that can emerge without this flag can look like:
# "clang : error : overriding '-mmacosx-version-min=11.0' option with '-target x86_64-apple-ios14.0-macabi' [-Werror,-Woverriding-t-option]"
if (PLATFORM_NAME STREQUAL "macabi")
  add_compile_options(-Wno-overriding-t-option)
  add_link_options(-Wno-overriding-t-option)
endif()

if(ENABLE_TESTS)
  # call enable_testing so we can add tests from subdirectories (e.g. test and src/java)
  # it applies recursively to all subdirectories
  enable_testing()
  if (USE_CUDA OR TEST_PHI2)
    add_compile_definitions(TEST_PHI2=1)
  else()
    add_compile_definitions(TEST_PHI2=0)
  endif()
endif()

if(ENABLE_TRACING)
  message(STATUS "Tracing is enabled.")
  add_compile_definitions(ORTGENAI_ENABLE_TRACING)
endif()

find_package(Threads REQUIRED)

if(WIN32)
  add_library(onnxruntime-genai SHARED ${generator_srcs} "${GENERATORS_ROOT}/dll/onnxruntime-genai.rc")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_INFO=\"${VERSION_INFO}\")
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MAJOR=${VERSION_MAJOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_MINOR=${VERSION_MINOR})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_PATCH=${VERSION_PATCH})
  target_compile_definitions(onnxruntime-genai PRIVATE VERSION_SUFFIX=${VERSION_SUFFIX})
  target_compile_definitions(onnxruntime-genai PRIVATE FILE_NAME=\"onnxruntime-genai.dll\")
else()
  add_library(onnxruntime-genai SHARED ${generator_srcs})
endif()

target_include_directories(onnxruntime-genai PRIVATE ${ORT_HEADER_DIR})
target_include_directories(onnxruntime-genai PRIVATE ${onnxruntime_extensions_SOURCE_DIR}/shared/api)
target_link_libraries(onnxruntime-genai PRIVATE onnxruntime_extensions)
target_link_directories(onnxruntime-genai PRIVATE ${ORT_LIB_DIR})
target_link_libraries(onnxruntime-genai PRIVATE Threads::Threads)

# Configure WebGPU support if enabled
if(USE_WEBGPU)
  message(STATUS "WebGPU support enabled")

  # Dawn WebGPU library paths
  set(DAWN_ROOT "C:/Users/jiajiaqin/workspace/dawn")

  # Set build directory based on CMAKE_BUILD_TYPE
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DAWN_BUILD_DIR "${DAWN_ROOT}/out/Debug")
    set(DAWN_LIB_SUBDIR "Debug")
    message(STATUS "Using Dawn Debug libraries")
  else()
    set(DAWN_BUILD_DIR "${DAWN_ROOT}/out/Release")
    set(DAWN_LIB_SUBDIR "Release")
    message(STATUS "Using Dawn Release libraries")
  endif()

  # Check if Dawn is available
  if(NOT EXISTS "${DAWN_BUILD_DIR}")
    message(FATAL_ERROR "Dawn build directory not found at ${DAWN_BUILD_DIR}. Please build Dawn first.")
  endif()

  # Dawn include directories
  target_include_directories(onnxruntime-genai PRIVATE
    "${DAWN_ROOT}/include"
    "${DAWN_BUILD_DIR}/gen/include"
  )

  # Use direct library paths - all Dawn/Tint/Abseil dependencies
  set(DAWN_LIBRARIES
    # Dawn libraries
    "${DAWN_BUILD_DIR}/src/dawn/common/${DAWN_LIB_SUBDIR}/dawn_common.lib"
    "${DAWN_BUILD_DIR}/src/dawn/${DAWN_LIB_SUBDIR}/dawn_proc.lib"
    "${DAWN_BUILD_DIR}/src/dawn/glfw/${DAWN_LIB_SUBDIR}/dawn_glfw.lib"
    "${DAWN_BUILD_DIR}/src/dawn/native/${DAWN_LIB_SUBDIR}/dawn_native.lib"
    "${DAWN_BUILD_DIR}/src/dawn/native/dawn_native_objects.dir/${DAWN_LIB_SUBDIR}/dawn_native_objects.lib"
    "${DAWN_BUILD_DIR}/src/dawn/platform/${DAWN_LIB_SUBDIR}/dawn_platform.lib"
    "${DAWN_BUILD_DIR}/src/dawn/utils/${DAWN_LIB_SUBDIR}/dawn_system_utils.lib"
    "${DAWN_BUILD_DIR}/src/dawn/utils/${DAWN_LIB_SUBDIR}/dawn_test_utils.lib"
    "${DAWN_BUILD_DIR}/src/dawn/utils/${DAWN_LIB_SUBDIR}/dawn_wgpu_utils.lib"
    "${DAWN_BUILD_DIR}/src/dawn/wire/${DAWN_LIB_SUBDIR}/dawn_wire.lib"

    # Tint API and core libraries
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_api.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_api_common.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_cmd_common.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_cmd_fuzz_ir_helpers.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_constant.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_intrinsic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_ir_analysis.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_ir_binary.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_ir_transform.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_ir_type.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_core_type.lib"

    # Tint WGSL support
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_ast.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_inspector.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_intrinsic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_program.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_reader.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_reader_lower.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_reader_parser.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_reader_program_to_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_resolver.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_sem.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_writer.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_writer_ast_printer.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_writer_ir_to_program.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_writer_raise.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_wgsl_writer_syntax_tree_printer.lib"

    # Tint SPIRV support
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_intrinsic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_reader.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_reader_ast_lower.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_reader_ast_parser.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_reader_common.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_reader_lower.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_reader_parser.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_type.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_validate.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_writer.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_writer_common.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_writer_helpers.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_writer_printer.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_spirv_writer_raise.lib"

    # Tint HLSL support
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_intrinsic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_type.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_validate.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_writer.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_writer_common.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_writer_helpers.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_writer_printer.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_hlsl_writer_raise.lib"

    # Tint GLSL support
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_glsl.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_glsl_intrinsic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_glsl_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_glsl_validate.lib"

    # Tint MSL support
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_msl.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_msl_intrinsic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_msl_ir.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_msl_ir_transform.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_lang_msl_type.lib"

    # Tint utility libraries
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_bytes.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_command.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_containers.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_diagnostic.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_file.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_ice.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_macros.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_math.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_memory.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_protos_ir_fuzz_proto.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_protos_ir_proto.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_rtti.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_strconv.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_symbol.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_system.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_text.lib"
    "${DAWN_BUILD_DIR}/src/tint/${DAWN_LIB_SUBDIR}/tint_utils_text_generator.lib"

    # Abseil base libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_base.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_log_severity.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_malloc_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_raw_logging_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_spinlock_wait.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_strerror.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/base/${DAWN_LIB_SUBDIR}/absl_throw_delegate.lib"

    # Abseil container libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/container/${DAWN_LIB_SUBDIR}/absl_hashtablez_sampler.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/container/${DAWN_LIB_SUBDIR}/absl_raw_hash_set.lib"

    # Abseil hash libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/hash/${DAWN_LIB_SUBDIR}/absl_city.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/hash/${DAWN_LIB_SUBDIR}/absl_hash.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/hash/${DAWN_LIB_SUBDIR}/absl_low_level_hash.lib"

    # Abseil debugging libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_debugging_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_demangle_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_examine_stack.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_failure_signal_handler.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_leak_check.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_stacktrace.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/debugging/${DAWN_LIB_SUBDIR}/absl_symbolize.lib"

    # Abseil string libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_cord.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_cordz_functions.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_cordz_handle.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_cordz_info.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_cordz_sample_token.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_cord_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_strings.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_strings_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_string_view.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/strings/${DAWN_LIB_SUBDIR}/absl_str_format_internal.lib"

    # Abseil synchronization libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/synchronization/${DAWN_LIB_SUBDIR}/absl_graphcycles_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/synchronization/${DAWN_LIB_SUBDIR}/absl_kernel_timeout_internal.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/synchronization/${DAWN_LIB_SUBDIR}/absl_synchronization.lib"

    # Abseil time libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/time/${DAWN_LIB_SUBDIR}/absl_civil_time.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/time/${DAWN_LIB_SUBDIR}/absl_time.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/time/${DAWN_LIB_SUBDIR}/absl_time_zone.lib"

    # Abseil status libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/status/${DAWN_LIB_SUBDIR}/absl_status.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/status/${DAWN_LIB_SUBDIR}/absl_statusor.lib"

    # Abseil numeric libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/numeric/${DAWN_LIB_SUBDIR}/absl_int128.lib"

    # Abseil types libraries
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/types/${DAWN_LIB_SUBDIR}/absl_bad_any_cast_impl.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/types/${DAWN_LIB_SUBDIR}/absl_bad_optional_access.lib"
    "${DAWN_BUILD_DIR}/third_party/abseil/absl/types/${DAWN_LIB_SUBDIR}/absl_bad_variant_access.lib"

    # SPIRV-Tools libraries
    "${DAWN_BUILD_DIR}/third_party/spirv-tools/source/${DAWN_LIB_SUBDIR}/SPIRV-Tools.lib"
    "${DAWN_BUILD_DIR}/third_party/spirv-tools/source/opt/${DAWN_LIB_SUBDIR}/SPIRV-Tools-opt.lib"

    # DXC libraries (DirectX Shader Compiler)
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/dxcompiler.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/dxclib.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/dxcvalidator.lib"

    # DXC LLVM core libraries
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMCore.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMSupport.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMAnalysis.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMBitReader.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMBitWriter.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMTransformUtils.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMInstCombine.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMScalarOpts.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMipo.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMVectorize.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMTarget.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMPasses.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMOption.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMProfileData.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMLinker.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMIRReader.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMAsmParser.lib"

    # DXC DXIL libraries
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDXIL.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxilContainer.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxilValidation.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxilHash.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxilCompression.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxilRootSignature.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxcSupport.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMDxcBindingTable.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMHLSL.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/LLVMMSSupport.lib"

    # DXC Clang libraries
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangBasic.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangLex.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangParse.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangAST.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangSema.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangAnalysis.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangEdit.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangRewrite.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangDriver.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangCodeGen.lib"
    "${DAWN_BUILD_DIR}/third_party/dxc/${DAWN_LIB_SUBDIR}/lib/clangFrontend.lib"
  )

  # Link all Dawn/Tint/Abseil libraries
  target_link_libraries(onnxruntime-genai PRIVATE ${DAWN_LIBRARIES})

  # Link Windows system libraries required by Dawn
  if(WIN32)
    target_link_libraries(onnxruntime-genai PRIVATE
      user32.lib
      dxguid.lib
      kernel32.lib
      gdi32.lib
      winspool.lib
      shell32.lib
      ole32.lib
      oleaut32.lib
      uuid.lib
      comdlg32.lib
      advapi32.lib
      onecore.lib
    )
  endif()

  target_compile_definitions(onnxruntime-genai PRIVATE USE_WEBGPU)
  list(LENGTH DAWN_LIBRARIES NUM_LIBRARIES)
  message(STATUS "WebGPU support configured with ${NUM_LIBRARIES} Dawn/Tint/Abseil libraries")

else()
  message(STATUS "WebGPU support disabled")
endif()

# The genai library itself is always embedded in the shared library
list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai>")

# we keep the shared libraries disconnected on Android as they will come from separate AARs and we don't want to force
# the ORT version to match in both.
if(CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Linux" OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND (NOT BUILD_APPLE_FRAMEWORK) AND (NOT MAC_CATALYST)))
  add_compile_definitions(_ORT_GENAI_USE_DLOPEN)
else()
  target_link_libraries(onnxruntime-genai PRIVATE ${ONNXRUNTIME_LIB})
endif()

if(APPLE)
target_link_libraries(onnxruntime-genai PRIVATE "-framework Foundation" "-framework CoreML")
endif()


# Build all source files using CUDA as a separate shared library we dynamically load at runtime
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  add_library(onnxruntime-genai-cuda SHARED ${generator_cudalib_srcs})
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${ORT_HEADER_DIR})
  target_include_directories(onnxruntime-genai-cuda PRIVATE ${GENERATORS_ROOT})
  target_link_libraries(onnxruntime-genai-cuda PRIVATE cublasLt cublas curand cufft cudart)
  set_target_properties(onnxruntime-genai-cuda PROPERTIES LINKER_LANGUAGE CUDA)
  add_dependencies(onnxruntime-genai onnxruntime-genai-cuda)
  source_group(TREE ${GENERATORS_ROOT}/cuda FILES ${generator_cudalib_srcs})
  list(APPEND ortgenai_embed_libs "$<TARGET_FILE:onnxruntime-genai-cuda>")
  if(APPLE)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${GENERATORS_ROOT}/cuda/exported_symbols.lst")
  elseif(UNIX)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${GENERATORS_ROOT}/cuda/version_script.lds -Xlinker --gc-sections")
  elseif(WIN32)
    set_property(TARGET onnxruntime-genai-cuda APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${GENERATORS_ROOT}/cuda/symbols.def")
  else()
    message(FATAL_ERROR "${target} unknown platform, need to specify shared library exports for it")
  endif()
endif()


if(USE_GUIDANCE)
  target_include_directories(onnxruntime-genai PUBLIC ${llguidance_SOURCE_DIR}/parser/)
  target_link_libraries(onnxruntime-genai PRIVATE llguidance)
  if (WIN32)
    # bcrypt is needed for the rust std lib
    target_link_libraries(onnxruntime-genai PRIVATE bcrypt)
  endif()
endif()

if(CMAKE_GENERATOR_TOOLSET MATCHES "Visual Studio")
  target_link_options(onnxruntime-genai PRIVATE "/CETCOMPAT")
  target_compile_options(onnxruntime-genai PRIVATE "/sdl")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set_target_properties(onnxruntime-genai PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(onnxruntime-genai PRIVATE dl)  # For dlopen & co
endif()

if(USE_DML)
  list(APPEND ortgenai_embed_libs "${D3D12_LIB_DIR}/D3D12Core.dll")
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${WIL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>/directx)
  target_include_directories(onnxruntime-genai PRIVATE $<TARGET_PROPERTY:${DIRECTX_HEADERS_TARGET},INTERFACE_INCLUDE_DIRECTORIES>)
  target_include_directories(onnxruntime-genai PRIVATE ${DML_HEADER_DIR})
  target_include_directories(onnxruntime-genai PRIVATE ${D3D12_HEADER_DIR})
  target_link_directories(onnxruntime-genai PRIVATE ${DML_LIB_DIR})
  target_link_directories(onnxruntime-genai PRIVATE ${D3D12_LIB_DIR})
  target_link_libraries(onnxruntime-genai PRIVATE d3d12.lib dxcore.lib dxguid.lib dxgi.lib)

  get_filename_component(PACKAGES_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps ABSOLUTE)
  set(DXC_PACKAGE_DIR ${PACKAGES_DIR}/Microsoft.Direct3D.DXC.1.7.2308.12)
  set(NUGET_CONFIG ${PROJECT_SOURCE_DIR}/nuget.config)
  set(PACKAGES_CONFIG ${PROJECT_SOURCE_DIR}/packages.config)

  add_custom_command(
    OUTPUT
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
    DEPENDS
    ${PACKAGES_CONFIG}
    ${NUGET_CONFIG}
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nuget/src/nuget restore ${PACKAGES_CONFIG} -PackagesDirectory ${PACKAGES_DIR} -ConfigFile ${NUGET_CONFIG}
    VERBATIM
  )

  add_custom_target(
    RESTORE_PACKAGES ALL
    DEPENDS
    ${DXC_PACKAGE_DIR}/build/native/bin/x64/dxc.exe
  )

  add_dependencies(RESTORE_PACKAGES nuget)
  add_dependencies(onnxruntime-genai RESTORE_PACKAGES)
endif()

if(ANDROID)
  # strip the binary if it's not a build with debug info
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_RELEASE -s)
  set_target_properties(onnxruntime-genai PROPERTIES LINK_FLAGS_MINSIZEREL -s)
endif()

if(ENABLE_TESTS)
  message("------------------Enabling tests------------------")
  add_subdirectory("${REPO_ROOT}/test")
endif()

if(ENABLE_PYTHON)
  message("------------------Enabling Python Wheel------------------")
  add_subdirectory("${SRC_ROOT}/python")
endif()

if (ENABLE_JAVA)
  message("------------------Enabling Java Jar------------------")
  add_subdirectory("${SRC_ROOT}/java")
endif()

if(ENABLE_MODEL_BENCHMARK)
  message("------------------Enabling model benchmark------------------")
  add_subdirectory("${REPO_ROOT}/benchmark/c")
endif()

# Have visual studio put all files into one single folder vs the default split of header files into a separate folder
source_group(TREE ${GENERATORS_ROOT} FILES ${generator_srcs})

include(cmake/package.cmake)
