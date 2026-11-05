// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>
#include <string>

#include <gtest/gtest.h>

#include "models/model.h"
#include "test_utils.h"

namespace Generators {
namespace {

struct CaptureTestState : State {
  using State::GraphIdForLength;
  using State::State;

  DeviceSpan<float> Run(int, DeviceSpan<int32_t>&, DeviceSpan<int32_t>) override {
    return {};
  }
};

class StateGraphCaptureTest : public testing::Test {
 protected:
  void SetUp() override {
    model_ = CreateModel(GetOrtEnv(), MODEL_PATH "engine/dummy-decoder");
    params_ = CreateGeneratorParams(*model_);
    params_->use_graph_capture = true;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<GeneratorParams> params_;
};

TEST_F(StateGraphCaptureTest, AdapterRevisionsHaveDistinctStableIds) {
  CaptureTestState state{*params_, *model_};
  const int base = state.GraphIdForLength(1, 0);
  ASSERT_GT(base, 0);
  EXPECT_EQ(state.GraphIdForLength(1, 0), base);

  // Exercise State's identity directly; Python tests cover real adapter activation and replay.
  state.adapter_names_.push_back("a");
  const int adapter_a = state.GraphIdForLength(1, 0);
  ASSERT_GT(adapter_a, 0);
  EXPECT_NE(adapter_a, base);
  EXPECT_EQ(state.GraphIdForLength(1, 0), adapter_a);

  state.adapter_names_.push_back("b");
  const int adapters_ab = state.GraphIdForLength(1, 0);
  ASSERT_GT(adapters_ab, 0);
  EXPECT_NE(adapters_ab, base);
  EXPECT_NE(adapters_ab, adapter_a);
  EXPECT_EQ(state.GraphIdForLength(1, 0), adapters_ab);

  const int other_length = state.GraphIdForLength(2, 0);
  const int other_variant = state.GraphIdForLength(1, 1);
  EXPECT_GT(other_length, 0);
  EXPECT_GT(other_variant, 0);
  EXPECT_NE(other_length, adapters_ab);
  EXPECT_NE(other_variant, adapters_ab);
  EXPECT_NE(other_length, other_variant);
  EXPECT_EQ(state.GraphIdForLength(2, 0), other_length);
  EXPECT_EQ(state.GraphIdForLength(1, 1), other_variant);
}

TEST_F(StateGraphCaptureTest, StatesSharingAModelNeverShareIds) {
  CaptureTestState first{*params_, *model_};
  std::set<int> retired;
  {
    CaptureTestState second{*params_, *model_};
    for (int revision = 0; revision < 3; ++revision) {
      const int first_id = first.GraphIdForLength(1, 0);
      const int second_id = second.GraphIdForLength(1, 0);
      EXPECT_GT(first_id, 0);
      EXPECT_GT(second_id, 0);
      EXPECT_NE(first_id, second_id);
      retired.insert(second_id);
      first.adapter_names_.push_back(std::to_string(revision));
      second.adapter_names_.push_back(std::to_string(revision));
    }
  }
  CaptureTestState fresh{*params_, *model_};
  const int fresh_id = fresh.GraphIdForLength(1, 0);
  EXPECT_GT(fresh_id, 0);
  EXPECT_FALSE(retired.contains(fresh_id));
  EXPECT_NE(fresh_id, first.GraphIdForLength(1, 0));
}

std::string capture_log;

struct ScopedCaptureLog {
  ScopedCaptureLog() : previous{g_log} {
    capture_log.clear();
    g_log.enabled = false;
    g_log.graph_capture = true;
    SetLogCallback([](const char* text, size_t length) { capture_log.append(text, length); });
  }

  ~ScopedCaptureLog() {
    SetLogCallback(nullptr);
    g_log = previous;
  }

  LogItems previous;
};

TEST_F(StateGraphCaptureTest, BudgetFallbackLogsOnceAndKeepsExistingIds) {
  ScopedCaptureLog log;
  CaptureTestState state{*params_, *model_};
  int last_id = -1;
  std::set<int> assigned;
  for (size_t revision = 0; revision < GraphAnnotationIds::kMaxCapturedShapes; ++revision) {
    last_id = state.GraphIdForLength(1, 0);
    ASSERT_GT(last_id, 0);
    EXPECT_TRUE(assigned.insert(last_id).second);
    if (revision + 1 < GraphAnnotationIds::kMaxCapturedShapes) {
      state.adapter_names_.push_back(std::to_string(revision));
    }
  }
  // An unknown key falls back; a previously assigned key can still replay.
  EXPECT_EQ(state.GraphIdForLength(2, 0), -1);
  EXPECT_TRUE(capture_log.empty());
  EXPECT_EQ(state.GraphIdForLength(1, 0), last_id);

  g_log.enabled = true;
  g_log.graph_capture = false;
  EXPECT_EQ(state.GraphIdForLength(2, 0), -1);
  EXPECT_TRUE(capture_log.empty());

  g_log.graph_capture = true;
  EXPECT_EQ(state.GraphIdForLength(2, 0), -1);
  EXPECT_NE(capture_log.find("capture budget exhausted"), std::string::npos);
  const auto first_message = capture_log;
  EXPECT_EQ(state.GraphIdForLength(1, 0), last_id);
  EXPECT_EQ(state.GraphIdForLength(2, 0), -1);
  state.adapter_names_.push_back("over_budget");
  EXPECT_EQ(state.GraphIdForLength(1, 0), -1);
  EXPECT_EQ(capture_log, first_message);
}

}  // namespace
}  // namespace Generators
