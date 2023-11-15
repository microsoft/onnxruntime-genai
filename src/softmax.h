#pragma once

namespace Generators {

void softmax(std::span<float> values);
void log_softmax(std::span<float> values);

}  // namespace Generators
