add_project_private_dependency(benchmark REQUIRED)

add_custom_target(bench)

function(ADD_SIMPLEX_BENCHMARK benchmark_name)
  set(target_name "${PROJECT_NAME}-benchmark-${benchmark_name}")
  add_executable(${target_name} ${benchmark_name}.cpp)
  target_link_libraries(${target_name} PRIVATE ${PROJECT_NAME})
  target_link_libraries(${target_name} PRIVATE benchmark::benchmark)

  add_dependencies(bench ${target_name})
endfunction()

add_simplex_benchmark(mujoco-humanoid)
add_simplex_benchmark(affine-transform)
add_simplex_benchmark(mujoco-humanoid-introspection)
