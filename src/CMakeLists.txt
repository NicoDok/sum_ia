# Note that headers are optional, and do not affect add_library, but they will not
# show up in IDEs unless they are listed in add_library.

# Optionally glob, but only for CMake 3.12 or later:
file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${Sum_IA_SOURCE_DIR}/include/mathlib/*.h")

# Make an automatic library - will be static or dynamic based on user setting
add_library(mathlib STATIC matrix.cpp vector.cpp ${HEADER_LIST})

# We need this directory, and users of our library will need it too
target_include_directories(mathlib PUBLIC ../include)

# All users of this library will need at least C++17
target_compile_features(mathlib PUBLIC cxx_std_17)

# IDEs should put the headers in a nice place
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
