set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

include_directories(
    "${CMAKE_SOURCE_DIR}/App/Include"
    "${CMAKE_SOURCE_DIR}/App/External"
)

file(GLOB_RECURSE APP_SRC "${CMAKE_SOURCE_DIR}/App/src/*.cpp")

add_executable(App
    ${APP_SRC}
    "${CMAKE_SOURCE_DIR}/App/Include/interface.hpp" #for automoc
    "${CMAKE_SOURCE_DIR}/App/Include/trainer.hpp" #for automoc
    "${CMAKE_SOURCE_DIR}/App/Include/ganTrainer.hpp" #for automoc
)

set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG")
target_compile_definitions(App PRIVATE EIGEN_NO_DEBUG)
target_compile_definitions(App PRIVATE EIGEN_DONT_PARALLELIZE=0)
set(ENV{OMP_NUM_THREADS} 4)
target_compile_definitions(App PRIVATE NDEBUG)
target_link_libraries(App PRIVATE Qt6::Widgets)