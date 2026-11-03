
#include <cuvs/cluster/gmm.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                                         \
  do {                                                                                        \
    cudaError_t e = (x);                                                                       \
    if (e != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__);   \
      exit(1);                                                                                 \
    }                                                                                          \
  } while (0)

static std::vector<float> read_f32(const std::string& path, size_t n)
{
  std::vector<float> v(n);
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  if (fread(v.data(), sizeof(float), n, f) != n) {
    fprintf(stderr, "short read on %s\n", path.c_str()); exit(1);
  }
  fclose(f);
  return v;
}

static void write_f32(const std::string& path, const std::vector<float>& v)
{
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot open %s for write\n", path.c_str()); exit(1); }
  fwrite(v.data(), sizeof(float), v.size(), f);
  fclose(f);
}

int main(int argc, char** argv)
{
  if (argc < 7) { fprintf(stderr, "usage: %s N d K iters in_dir out_prefix\n", argv[0]); return 2; }
  int64_t     N     = atoll(argv[1]);
  int64_t     d     = atoll(argv[2]);
  int         K     = atoi(argv[3]);
  int         iters = atoi(argv[4]);
  std::string in    = argv[5];
  std::string outp  = argv[6];

  auto hX = read_f32(in + "/X.f32", (size_t)N * d);
  auto hW = read_f32(in + "/w0.f32", (size_t)K);
  auto hM = read_f32(in + "/m0.f32", (size_t)K * d);
  auto hC = read_f32(in + "/v0.f32", (size_t)K * d);   // DIAG: variances, flat K*d

  float *dX, *dW, *dM, *dC, *dPC, *dP;
  int*   dL;
  CUDA_CHECK(cudaMalloc(&dX, sizeof(float) * N * d));
  CUDA_CHECK(cudaMalloc(&dW, sizeof(float) * K));
  CUDA_CHECK(cudaMalloc(&dM, sizeof(float) * K * d));
  CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * K * d));
  CUDA_CHECK(cudaMalloc(&dPC, sizeof(float) * K * d));
  CUDA_CHECK(cudaMalloc(&dP, sizeof(float) * K * d));
  CUDA_CHECK(cudaMalloc(&dL, sizeof(int) * N));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), sizeof(float) * N * d, cudaMemcpyHostToDevice));

  raft::resources             handle;
  cuvs::cluster::gmm::params  p;
  p.n_components = K;
  p.cov_type     = cuvs::cluster::gmm::covariance_type::DIAG;
  p.max_iter     = iters;
  p.tol          = 0.0;   // |dlb| < 0 is never true -> exactly max_iter, no early exit
  p.reg_covar    = 1e-6;
  p.n_init       = 1;
  p.seed         = 0;

  auto Xv  = raft::make_device_matrix_view<const float, int64_t>(dX, N, d);
  auto Wv  = raft::make_device_vector_view<float, int64_t>(dW, K);
  auto Mv  = raft::make_device_matrix_view<float, int64_t>(dM, K, d);
  auto Cv  = raft::make_device_vector_view<float, int64_t>(dC, (int64_t)K * d);
  auto PCv = raft::make_device_vector_view<float, int64_t>(dPC, (int64_t)K * d);
  auto Pv  = raft::make_device_vector_view<float, int64_t>(dP, (int64_t)K * d);
  auto Lv  = raft::make_device_vector_view<int, int64_t>(dL, N);

  float lb;
  int   n_iter;
  bool  converged;

  // warm_start = true: the incoming weights/means/covariances ARE the initialization,
  // and n_init is ignored, so the init matches every other arm exactly.
  auto run = [&]() {
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), sizeof(float) * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dM, hM.data(), sizeof(float) * K * d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC.data(), sizeof(float) * K * d, cudaMemcpyHostToDevice));
    cuvs::cluster::gmm::fit(handle, p, Xv, Wv, Mv, Cv, PCv, Pv, Lv,
                            raft::make_host_scalar_view(&lb),
                            raft::make_host_scalar_view(&n_iter),
                            raft::make_host_scalar_view(&converged),
                            /* warm_start */ true);
    CUDA_CHECK(cudaDeviceSynchronize());
  };

  run();                                        // warmup
  const int reps = 5;
  std::vector<double> ms;
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run();
    auto t1 = std::chrono::high_resolution_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  std::vector<float> oW(K), oM((size_t)K * d), oC((size_t)K * d);
  CUDA_CHECK(cudaMemcpy(oW.data(), dW, sizeof(float) * K, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(oM.data(), dM, sizeof(float) * K * d, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(oC.data(), dC, sizeof(float) * K * d, cudaMemcpyDeviceToHost));
  write_f32(outp + "w.f32", oW);
  write_f32(outp + "mu.f32", oM);
  write_f32(outp + "va.f32", oC);

  double best = *std::min_element(ms.begin(), ms.end());
  printf("{\"arm\": \"cuvs\", \"total_ms\": %.6f, \"all_ms\": [%.6f, %.6f, %.6f, %.6f, %.6f], "
         "\"iters\": %d, \"peak_mb\": null, \"lower_bound\": %.6f, \"converged\": %d, "
         "\"setup_note\": \"cuVS main @ nightly 26.12.0a14 (GMM is unreleased); "
         "one-thread-per-sample SIMT E-step, no tensor cores; total_ms is the MIN of 5 reps because repeated fit() calls drift upward on allocator state\", "
         "\"versions\": {\"libcuvs\": \"26.12.0a14.post260913025751\"}}\n",
         best, ms[0], ms[1], ms[2], ms[3], ms[4], n_iter, lb, (int)converged);
  return 0;
}
