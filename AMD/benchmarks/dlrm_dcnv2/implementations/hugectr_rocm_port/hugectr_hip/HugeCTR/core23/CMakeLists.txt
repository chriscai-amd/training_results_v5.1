# HIP port: HugeCTR core23 base library
cmake_minimum_required(VERSION 3.21)

set(core23_src
  allocator_factory.cpp
  allocator_params.cpp
  buffer.cpp
  buffer_client.cpp
  buffer_channel.cpp
  buffer_channel_helpers.cpp
  buffer_factory.cpp
  buffer_params.cpp
  device.cpp
  device_guard.cpp
  device_type.cpp
  data_type.cpp
  offsetted_buffer.cpp
  low_level_primitives.cpp
  low_level_primitives.cu
  mpi_init_service.cpp
  details/simple_cuda_allocator.cpp
  details/managed_cuda_allocator.cpp
  details/low_level_cuda_allocator.cpp
  details/pool_cuda_allocator.cpp
  details/pinned_host_allocator.cpp
  details/new_delete_allocator.cpp
  details/unitary_buffer.cpp
  details/confederal_buffer.cpp
  details/tensor_impl.cpp
  details/tensor_helpers.cpp
  details/host_launch_helpers.cpp
  tensor.cpp
  tensor_operations.cpp
  kernel_params.cpp
  shape.cpp
  logger.cpp
)

# Mark .cu files as HIP language
foreach(f ${core23_src})
  if(f MATCHES "\\.cu$")
    set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

add_library(hugectr_core23 SHARED ${core23_src})
target_link_libraries(hugectr_core23
  PUBLIC
    hip::host
    roc::rocrand
    hip::hiprand
)
target_compile_features(hugectr_core23 PRIVATE cxx_std_17)
