project(final_project)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
cmake_minimum_required(VERSION 2.8.12)

# Find external modules
find_package(OpenCV 3 REQUIRED)


# Compile options
if((CMAKE_CXX_COMPILER_ID MATCHES "Clang") OR (CMAKE_CXX_COMPILER_ID MATCHES "GNU"))
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wcast-qual -Wcast-align)
endif()
# Target executable
add_executable(COLOR "color.cpp")
add_executable(TEST "test.cpp")
add_executable(Marker2 "marker2.cpp")

target_link_libraries(TEST ${OpenCV_LIBS})
target_link_libraries(COLOR ${OpenCV_LIBS})
target_link_libraries(Marker2 ${OpenCV_LIBS})
