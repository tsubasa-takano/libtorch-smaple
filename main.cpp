#include <torch/script.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <vector>

int main() {
    std::string model_path = "../models/uniformerv2_tiny.pt";

    std::cout << "Looking for model at: " << model_path << std::endl;

    if (!std::filesystem::exists(model_path)) {
        std::cerr << "モデルファイルが見つかりません: " << model_path << std::endl;
        return -1;
    }

    try {
        constexpr int BATCH_SIZE = 1;
        constexpr int INPUT_FRAMES = 5;
        constexpr int IMAGE_HEIGHT = 120;
        constexpr int IMAGE_WIDTH = 180;

        // JITコンパイル最適化オプションの追加
        torch::jit::FusionStrategy fusion_strategy;
        fusion_strategy.emplace_back(torch::jit::FusionBehavior::DYNAMIC, 1);
        torch::jit::setFusionStrategy(fusion_strategy);

        std::cout << "Loading model from: " << model_path << std::endl;

        // モデルロード時に最適化オプションを使用
        torch::jit::script::Module model;
        {
            torch::NoGradGuard no_grad; // 勾配計算不要のモードに設定
            model = torch::jit::load(model_path, torch::kCUDA);
        }

        // デバイスの設定
        torch::Device device(torch::kCUDA);
        model.eval(); // 評価モードに設定

        // cuDNN最適化設定
        at::globalContext().setBenchmarkCuDNN(true);
        at::globalContext().setDeterministicCuDNN(false);
        at::globalContext().setAllowTF32CuDNN(true);  // cuDNN演算でTF32を使用
        at::globalContext().setAllowTF32CuBLAS(true); // cuBLAS演算でTF32を使用

        // メモリをピン留めしてCPU-GPU間の転送を高速化
        at::globalContext().setBenchmarkCuDNN(true);

        // 入力テンソルを作成
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device)
            .requires_grad(false);

        // contiguousメモリレイアウトで作成
        auto input_tensor = torch::randn({BATCH_SIZE, 3, INPUT_FRAMES, IMAGE_HEIGHT, IMAGE_WIDTH}, options).contiguous();

        // ウォームアップ実行
        constexpr int NUM_WARMUP = 10;
        std::cout << "Running warmup..." << std::endl;
        {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < NUM_WARMUP; ++i) {
                auto output = model.forward({input_tensor});
            }
        }

        // 推論時間の計測
        constexpr int NUM_RUNS = 1000;
        std::vector<double> inference_times;
        inference_times.reserve(NUM_RUNS); // メモリを事前確保

        std::cout << "Running inference..." << std::endl;
        {
            torch::NoGradGuard no_grad;
            for (int i = 0; i < NUM_RUNS; ++i) {
                // CUDA同期して計測を正確に行う
                auto start = std::chrono::high_resolution_clock::now();

                auto output = model.forward({input_tensor});

                auto end = std::chrono::high_resolution_clock::now();

                std::chrono::duration<double, std::milli> elapsed = end - start;
                inference_times.push_back(elapsed.count());
            }
        }

        // 統計計算
        double sum = 0.0, sum_sq = 0.0;
        for (const auto& time : inference_times) {
            sum += time;
            sum_sq += time * time;
        }

        const double avg_time = sum / NUM_RUNS;
        const double std_time = std::sqrt(sum_sq / NUM_RUNS - avg_time * avg_time);

        std::cout << "Average Inference Time: " << avg_time << " ms ± " << std_time << " ms" << std::endl;

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}