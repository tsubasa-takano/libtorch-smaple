cmake_minimum_required(VERSION 3.28.3)
project(libtroch_sample)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 17)

# CUDA アーキテクチャを明示的に設定
set(CMAKE_CUDA_ARCHITECTURES 80 86)  # 使用している GPU に合わせて調整

# 最適化フラグの設定
# -O3: 最も高度な最適化レベル。コード実行速度を最大化するためにインライン展開やループアンロールなどを実施
# -march=native: コンパイルを実行しているCPUのアーキテクチャに合わせた命令セット(AVX, SSE, FMAなど)を使用
# -mtune=native: 現在のCPUに合わせて命令スケジューリングなどを最適化。-marchと組み合わせて使用すると効果的
# これらのフラグによりCPU側の処理効率が向上し、Torch/CUDA操作を含む計算負荷の高いコードの実行速度が向上する
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")


add_executable(${PROJECT_NAME} main.cpp)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# libtorch
set(Torch_DIR "./oss/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
