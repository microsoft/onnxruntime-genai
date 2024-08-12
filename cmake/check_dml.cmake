# Checking if DML is supported

if(USE_DML)
  if(WIN32)
    add_compile_definitions(USE_DML=1)
    add_compile_definitions(NOMINMAX)
    add_compile_definitions(DML_TARGET_VERSION_USE_LATEST)

    file(GLOB dml_srcs CONFIGURE_DEPENDS
      "${PROJECT_SOURCE_DIR}/src/dml/*.h"
      "${PROJECT_SOURCE_DIR}/src/dml/*.cpp"
    )

    list(APPEND generator_srcs ${dml_srcs})
  else()
    message(FATAL_ERROR "USE_DML is ON but this isn't windows.")
  endif()
else()
  add_compile_definitions(USE_DML=0)
endif()