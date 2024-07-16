#include <iostream>
#include <vector>
#include "chrono"
#include "server/engine_utils.h"

#include "engine.h"
namespace engine {

OgaEngine::OgaEngine(const char* config_path) {
  // Load the configuration file
  // Initialize the engine

  std::cout << "Creating model..." << std::endl;
  model_ = Generators::CreateModel(Generators::GetOrtEnv(), config_path) std::cout
           << "Creating tokenizer..." << std::endl;
  tokenizer_ = model_->CreateTokenizer();
}

std::vector<const char*> OgaEngine::Schedule() {
  // Schedule the requests
  // Return the scheduled requests

  std::vector<const char*> scheduled_prompts;
  for (int i = 0; i < kMaxBatchSize; i++) {
    if (unscheduled_prompts_.empty()) {
      break;
    }
    scheduled_prompts.push_back(unscheduled_prompts_.front());
    unscheduled_prompts_.pop();
  }
  return scheduled_prompts;
}

std::vector<std::string> OgaEngine::Generate(std::vector<const char*> prompts) {
  // Generate the output for the scheduled requests
  // Print the output

  for (const char* prompt : prompts) {
    unscheduled_prompts_.push(prompt);
  }

  std::vector<std::string> outputs(prompts.size());

  while (!unscheduled_prompts_.empty()) {
    std::vector<const char*> scheduled_prompts = Schedule();
    auto sequences = OgaSequences::Create();

    tokenizer_->EncodeBatch(scheduled_prompts, *sequences);

    auto params = OgaGeneratorParams::Create(*model_);
    params->SetSearchOption("max_length", 200);
    params->SetInputSequences(*sequences);

    auto output_sequences = model_->Generate(*params);

    auto out_strings = tokenizer_->DecodeBatch(std::move(output_sequences));
    for (auto out_string : out_strings) {
      outputs.push_back(std::string(out_string));
    }
  }
  return outputs;
}



void OgaEngine::AddRequest(std::string request_id, std::string inputs,
                           SamplingParams params, float arrival_time) {
  if (arrival_time == 0) {
    arrival_time = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  }

  std::vector<int32_t> token_ids = tokenizer_->Encode(inputs.c_str());

  // TODO: get block_size
  Sequence seq{seq_count_, LLMInputs{token_ids, inputs}, 1024,
               tokenizer_->GetEosTokenId()};
  seq_count_++;

  SequenceGroup seq_group{request_id, {seq}, arrival_time, params, {}, nullptr};

  scheduler_->AddSeqGroup(std::move(seq_group));
}
 
std::vector<RequestOutput> OgaEngine::Step() {
  ScheduleResult schedule_result = scheduler_.Schedule();
  // build model executor input

  // process model outputs

  return {};
}

}  // namespace engine