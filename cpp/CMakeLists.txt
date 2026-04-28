cmake_minimum_required(VERSION 3.15)
project(AGVPerceptionClient CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# gRPC
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)
pkg_check_modules(PROTOBUF REQUIRED protobuf)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GRPC_INCLUDE_DIRS}
    ${PROTOBUF_INCLUDE_DIRS}
)

# 源文件
set(SOURCES
    perception_client.cpp
    logistics.grpc.pb.cc
    logistics.pb.cc
)

# 可执行文件
add_executable(agv_perception_client ${SOURCES})

# 链接库
target_link_libraries(agv_perception_client
    ${GRPC_LIBRARIES}
    ${PROTOBUF_LIBRARIES}
    pthread
)

# 编译Proto
find_program(PROTOC protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

add_custom_command(OUTPUT logistics.grpc.pb.cc logistics.pb.cc
    COMMAND ${PROTOC} --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
            logistics.proto
    DEPENDS ${PROTOC} logistics.proto
)

add_custom_target(protoc_build ALL DEPENDS logistics.grpc.pb.cc logistics.pb.cc)

add_dependencies(agv_perception_client protoc_build)
