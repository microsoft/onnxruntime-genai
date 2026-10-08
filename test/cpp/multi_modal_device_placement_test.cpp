// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Each state allocates against the device of the session it drives, never the decoder's
// (see SessionCanAccess).

#include <memory>

#include <gtest/gtest.h>

#include "models/model.h"

namespace {

struct PlacementTestModel : Generators::Model {
  PlacementTestModel() : Model{std::make_unique<Generators::Config>()} {}

  std::unique_ptr<Generators::State> CreateState(
      Generators::DeviceSpan<int32_t>, const Generators::GeneratorParams&) const override {
    return nullptr;
  }
};

struct PlacementTestState : Generators::State {
  PlacementTestState(const Generators::GeneratorParams& params, const Generators::Model& model,
                     Generators::DeviceInterface* session_device)
      : State{params, model, session_device} {}

  Generators::DeviceSpan<float> Run(int, Generators::DeviceSpan<int32_t>&,
                                    Generators::DeviceSpan<int32_t>) override {
    return {};
  }
};

TEST(MultiModalDevicePlacementTests, SessionCanAccessOnlyHostOrItsOwnMemory) {
  auto& cpu = *Generators::GetDeviceInterface(Generators::DeviceType::CPU);
  auto& gpu = *Generators::GetDeviceInterface(Generators::DeviceType::WEBGPU);

  EXPECT_TRUE(Generators::SessionCanAccess(cpu, cpu));
  EXPECT_TRUE(Generators::SessionCanAccess(gpu, gpu));
  // Host memory is always reachable: ORT copies it to and from the session's own device.
  EXPECT_TRUE(Generators::SessionCanAccess(gpu, cpu));
  // A CPU-EP session has no data transfer for device memory, which is the case that corrupts
  // the heap when the buffer is bound anyway.
  EXPECT_FALSE(Generators::SessionCanAccess(cpu, gpu));
}

TEST(MultiModalDevicePlacementTests, StateDefaultsToTheModelDevices) {
  auto model = std::make_shared<PlacementTestModel>();
  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  // A CPU model has p_device_inputs_ == p_device_, which would hide a state that picks the wrong one.
  model->p_device_inputs_ = Generators::GetDeviceInterface(Generators::DeviceType::WEBGPU);

  PlacementTestState state{*params, *model, nullptr};

  EXPECT_EQ(state.p_session_device_, model->p_device_);
  EXPECT_EQ(state.p_session_device_inputs_, model->p_device_inputs_);
}

TEST(MultiModalDevicePlacementTests, StateOnAnotherDeviceKeepsItsOwnInputsDevice) {
  auto model = std::make_shared<PlacementTestModel>();
  auto params = std::make_shared<Generators::GeneratorParams>(*model);
  auto* elsewhere = Generators::GetDeviceInterface(Generators::DeviceType::WEBGPU);
  ASSERT_NE(elsewhere, model->p_device_);

  PlacementTestState state{*params, *model, elsewhere};

  EXPECT_EQ(state.p_session_device_, elsewhere);
  // p_device_inputs_ was picked for the *model's* device, so a session that runs somewhere else
  // must not allocate its inputs from it.
  EXPECT_EQ(state.p_session_device_inputs_, elsewhere);
}

}  // namespace
