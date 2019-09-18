project(matmult-performance)

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_executable(opencl-test ocl.cpp)
target_link_libraries(opencl-test OpenCL::OpenCL)

add_executable(matmult-test mmult.cpp)
target_link_libraries(matmult-test Threads::Threads OpenMP::OpenMP_CXX)
