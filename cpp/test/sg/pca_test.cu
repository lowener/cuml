/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuml/decomposition/params.hpp>
#include <cuml/decomposition/pca.hpp>
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>
#include <test_utils.h>
#include <vector>

namespace ML {

template <typename T>
struct PcaInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  int n_comp;
  int len2;
  int n_row2;
  int n_col2;
  int n_comp2;
  unsigned long long int seed;
  int algo;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PcaInputs<T>& dims)
{
  return os;
}

template <typename T>
class PcaTest : public ::testing::TestWithParam<PcaInputs<T>> {
 public:
  PcaTest()
    : params(::testing::TestWithParam<PcaInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      explained_vars(params.n_comp, stream),
      explained_vars_ref(params.n_comp, stream),
      components(params.n_col * params.n_comp, stream),
      components_ref(params.n_col * params.n_comp, stream),
      trans_data(params.n_row * params.n_comp, stream),
      trans_data_ref(params.n_row * params.n_comp, stream),
      data(params.len, stream),
      data_back(params.len, stream),
      data_back_ref(params.len, stream),
      data2(params.len2, stream),
      data2_back(params.len2, stream)
  {
    basicTest();
    advancedTest();
  }

 protected:
  void basicTest()
  {
    int len = params.len;
    int n_comp = params.n_comp;
    int len_comp = params.n_col * n_comp;

    std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0, 6.0, 6.0, 2.0, 1.0, 0.0, 6.0};
    data_h.resize(len);
    raft::update_device(data.data(), data_h.data(), len, stream);

    std::vector<T> trans_data_ref_h = {-2.3336391 , -3.1228137 ,  1.5089133 ,  3.9475393, -1.1370366 , -0.17664827,  3.4408271 , -2.1271422};

    trans_data_ref_h.resize(params.n_row * n_comp);
    raft::update_device(trans_data_ref.data(), trans_data_ref_h.data(), params.n_row * n_comp, stream);

    std::vector<T> data_back_ref_h = {1.4951986 , 1.5341609 , 5.0751295 , 3.895511, 1.6396616 , 1.3389738 , 5.9453316 , 6.076033, 2.105885  , 0.9003931 , 0.01606417, 5.977658 };
    data_back_ref_h.resize(len);
    raft::update_device(data_back_ref.data(), data_back_ref_h.data(), len, stream);

    rmm::device_uvector<T> explained_var_ratio(n_comp, stream);
    rmm::device_uvector<T> singular_vals(n_comp, stream);
    rmm::device_uvector<T> mean(params.n_col, stream);
    rmm::device_uvector<T> noise_vars(1, stream);

    std::vector<T> components_ref_h = {0.44635436,  0.40734962, 0.7546988 ,  0.30706465, 0.48083007, -0.8601033};

    components_ref_h.resize(len_comp);
    std::vector<T> explained_vars_ref_h = {11.019241 ,  5.8960257};
    explained_vars_ref_h.resize(n_comp);

    raft::update_device(components_ref.data(), components_ref_h.data(), len_comp, stream);
    raft::update_device(
      explained_vars_ref.data(), explained_vars_ref_h.data(), n_comp, stream);

    paramsPCA prms;
    prms.n_cols       = params.n_col;
    prms.n_rows       = params.n_row;
    prms.n_components = params.n_comp;
    prms.whiten       = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;
    else
      prms.algorithm = solver::R_SVD;

    pcaFit(handle,
           data.data(),
           components.data(),
           explained_vars.data(),
           explained_var_ratio.data(),
           singular_vals.data(),
           mean.data(),
           noise_vars.data(),
           prms);
    pcaTransform(handle,
                 data.data(),
                 components.data(),
                 trans_data.data(),
                 singular_vals.data(),
                 mean.data(),
                 prms);
    pcaInverseTransform(handle,
                        trans_data.data(),
                        components.data(),
                        singular_vals.data(),
                        mean.data(),
                        data_back.data(),
                        prms);
  }

  void advancedTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.len2;
    r.uniform(data2.data(), len, T(-1.0), T(1.0), stream);

    paramsPCA prms;
    prms.n_cols       = params.n_col2;
    prms.n_rows       = params.n_row2;
    prms.n_components = params.n_col2;
    prms.whiten       = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;
    else {
      // Skip this test for R_SVD because n_components can't be equal to n_col
      raft::copy(data2_back.data(), data2.data(), len, stream);
      return;
    }

    rmm::device_uvector<T> data2_trans(prms.n_rows * prms.n_cols, stream);

    int len_comp = prms.n_cols * prms.n_components;
    rmm::device_uvector<T> components2(len_comp, stream);
    rmm::device_uvector<T> explained_vars2(prms.n_components, stream);
    rmm::device_uvector<T> explained_var_ratio2(prms.n_components, stream);
    rmm::device_uvector<T> singular_vals2(prms.n_components, stream);
    rmm::device_uvector<T> mean2(prms.n_cols, stream);
    rmm::device_uvector<T> noise_vars2(1, stream);

    pcaFitTransform(handle,
                    data2.data(),
                    data2_trans.data(),
                    components2.data(),
                    explained_vars2.data(),
                    explained_var_ratio2.data(),
                    singular_vals2.data(),
                    mean2.data(),
                    noise_vars2.data(),
                    prms);

    pcaInverseTransform(handle,
                        data2_trans.data(),
                        components2.data(),
                        singular_vals2.data(),
                        mean2.data(),
                        data2_back.data(),
                        prms);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream = 0;

  PcaInputs<T> params;

  rmm::device_uvector<T> explained_vars, explained_vars_ref, components, components_ref, trans_data,
    trans_data_ref, data, data_back, data_back_ref, data2, data2_back;
};

const std::vector<PcaInputs<float>> inputsf2 = {
  {0.01f, 4 * 3, 4, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 4 * 3, 4, 3, 2, 256 * 32, 256, 32, 1234ULL, 1},
  {0.01f, 4 * 3, 4, 3, 2, 256 * 32, 256, 32, 1234ULL, 2}};

const std::vector<PcaInputs<double>> inputsd2 = {
  {0.01, 4 * 3, 4, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01, 4 * 3, 4, 3, 2, 256 * 32, 256, 32, 1234ULL, 1},
  {0.01, 4 * 3, 4, 3, 2, 256 * 32, 256, 32, 1234ULL, 2}};

typedef PcaTest<float> PcaTestValF;
TEST_P(PcaTestValF, Result)
{
  ASSERT_TRUE(devArrMatch(explained_vars.data(),
                          explained_vars_ref.data(),
                          params.n_comp,
                          raft::CompareApproxAbs<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestValD;
TEST_P(PcaTestValD, Result)
{
  ASSERT_TRUE(devArrMatch(explained_vars.data(),
                          explained_vars_ref.data(),
                          params.n_comp,
                          raft::CompareApproxAbs<double>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<float> PcaTestLeftVecF;
TEST_P(PcaTestLeftVecF, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_comp),
                          raft::CompareApproxAbs<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestLeftVecD;
TEST_P(PcaTestLeftVecD, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_comp),
                          raft::CompareApproxAbs<double>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<float> PcaTestTransDataF;
TEST_P(PcaTestTransDataF, Result)
{
  ASSERT_TRUE(devArrMatch(trans_data.data(),
                          trans_data_ref.data(),
                          (params.n_row * params.n_comp),
                          raft::CompareApproxAbs<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestTransDataD;
TEST_P(PcaTestTransDataD, Result)
{
  ASSERT_TRUE(devArrMatch(trans_data.data(),
                          trans_data_ref.data(),
                          (params.n_row * params.n_comp),
                          raft::CompareApproxAbs<double>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<float> PcaTestDataVecSmallF;
TEST_P(PcaTestDataVecSmallF, Result)
{
  ASSERT_TRUE(devArrMatch(data_back.data(),
                          data_back_ref.data(),
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestDataVecSmallD;
TEST_P(PcaTestDataVecSmallD, Result)
{
  ASSERT_TRUE(devArrMatch(data_back.data(),
                          data_back_ref.data(),
                          (params.n_row * params.n_col),
                          raft::CompareApproxAbs<double>(params.tolerance),
                          handle.get_stream()));
}

// FIXME: These tests are disabled due to driver 418+ making them fail:
// https://github.com/rapidsai/cuml/issues/379
typedef PcaTest<float> PcaTestDataVecF;
TEST_P(PcaTestDataVecF, Result)
{
  ASSERT_TRUE(devArrMatch(data2.data(),
                          data2_back.data(),
                          (params.n_row2 * params.n_col2),
                          raft::CompareApproxAbs<float>(params.tolerance),
                          handle.get_stream()));
}

typedef PcaTest<double> PcaTestDataVecD;
TEST_P(PcaTestDataVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data2.data(),
                                data2_back.data(),
                                (params.n_row2 * params.n_col2),
                                raft::CompareApproxAbs<double>(params.tolerance),
                                handle.get_stream()));
}

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace ML
