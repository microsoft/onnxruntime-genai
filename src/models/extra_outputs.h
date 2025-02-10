#pragma once

namespace Generators {

struct ExtraOutputs {
public:
    ExtraOutputs(State& state);
    void Add();
    void Update();
    OrtValue* GetOutput(const char* name);

private:
    State& state_;
    // manage output ortvalues not specified in output_names_
    std::unordered_map<std::string, std::unique_ptr<OrtValue>> output_ortvalue_store_;
    std::vector<std::string> all_output_names_;  // keep output strings in scope
    size_t extra_outputs_start_{std::numeric_limits<size_t>::max()};
};


}  // namespace Generators