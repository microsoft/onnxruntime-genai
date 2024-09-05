#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <span>
#include "chrono"
#include "engine_utils.h"
#include "scheduler.h"

#include "engine.h"
namespace Generators {

OgaEngine::OgaEngine(const char* config_path) {
  // Load the configuration file
  // Initialize the engine

  std::cout << "Creating model..." << std::endl;
  model_ = Generators::CreateModel(Generators::GetOrtEnv(), config_path);
  std::cout << "Creating tokenizer..." << std::endl;
  tokenizer_ = model_->CreateTokenizer();
  CacheOptions cache_config{
      model_->config_->model.decoder.num_hidden_layers,
      256,
      model_->config_->model.decoder.num_key_value_heads,
      model_->config_->model.decoder.head_size,
      std::nullopt,
      64,
      std::nullopt,
  };
  block_size_ = cache_config.block_size_;
  model_runner_ = std::make_unique<ModelRunner>(model_, cache_config);
  SchedulerConfig scheduler_config;
  scheduler_config.max_model_len = model_->config_->model.context_length;
  auto block_manager = std::make_unique<CacheManager>(
      cache_config, &model_->allocator_cpu_, model_->allocator_device_);
  scheduler_ =
      std::make_unique<Scheduler>(scheduler_config, std::move(block_manager));
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

void OgaEngine::AddRequest(std::string request_id, const std::string& inputs,
                           SamplingParams params, float arrival_time) {
  if (arrival_time == 0) {
    arrival_time = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  }

  std::vector<int32_t> token_ids = tokenizer_->Encode(inputs.c_str());
  auto llm_inputs = LLMInputs{token_ids, inputs};
  std::cout << "create sequence start\n";
  // TODO: get block_size
  Sequence seq{seq_count_++, llm_inputs, block_size_,
               model_->config_->model.eos_token_id};
  std::cout << "finish sequence start\n";

  std::vector<Sequence> seqs{seq};
  std::vector<float> embeddings;

  SequenceGroup seq_group{request_id, seqs, arrival_time,
                          params, embeddings, nullptr};

  scheduler_->AddSeqGroup(std::move(seq_group));
}

std::vector<RequestOutput> OgaEngine::Step() {
  auto [seq_group_metadatas, scheduler_outputs] = scheduler_->Schedule();

  if (seq_group_metadatas.empty()) {
    return {};
  }

  std::cout << "Scheduled " << scheduler_outputs.scheduled_seq_groups.size()
            << " seq groups" << std::endl;

  std::cout << "seq_group_metadatas.size() = " << seq_group_metadatas.size()
            << std::endl;

  std::cout << "seq_group_metadatas[0].seq_data.size() = "
            << seq_group_metadatas.at(0).seq_data.size() << std::endl;

  for (const auto& [id, seq_data] : seq_group_metadatas.at(0).seq_data) {
    std::cout << "seq_data.GetLen() = " << seq_data.GetLen() << std::endl;
  }
  std::cout << "seq_group_metadatas.at(0).block_tables.size() = "
            << seq_group_metadatas.at(0).block_tables.size() << std::endl;
  for (const auto& [key, value] : seq_group_metadatas.at(0).block_tables) {
    std::cout << "key = " << key << std::endl;
    std::cout << "value.size() = " << value.size();
    for (const auto& v : value) {
      std::cout << v << " ";
    }
      std::cout << std::endl;
  }
  std::cout << "seq_group_metadatas.at(0).computed_block_nums.size() = "
            << seq_group_metadatas.at(0).computed_block_nums.size()
            << std::endl;

  // build model executor input
  ExecuteModelRequest model_req{seq_group_metadatas,
                                scheduler_outputs.blocks_to_swap_in,
                                scheduler_outputs.blocks_to_copy,
                                scheduler_outputs.blocks_to_copy,
                                scheduler_outputs.num_lookahead_slots,
                                scheduler_outputs.running_queue_size};
  auto outputs = model_runner_->ExecuteModel(model_req);
  std::cout << "outputs.size() = " << outputs.size() << std::endl;
  // process model outputs
  float now = std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

  auto& scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups;

  size_t queue_offset = scheduler_->GetRunningSize() - scheduled_seq_groups.size();
  for (int i = 0; i < scheduled_seq_groups.size(); i++) {
    auto& seq_group = scheduled_seq_groups[i].seq_group;
    auto& queued_seq_group = scheduler_->GetRunning(queue_offset + i);
    seq_group.UpdateNumComputedTokens(scheduled_seq_groups[i].token_chunk_size);
    queued_seq_group.UpdateNumComputedTokens(scheduled_seq_groups[i].token_chunk_size);

    auto& seq_group_metadata = seq_group_metadatas[i];
    // process output
    if (seq_group_metadata.do_sample) {
      auto& samples = outputs[i].samples;
      auto parent_seqs = seq_group.GetSeqs(SequenceStatus::kRunning);
      std::unordered_map<int, std::vector<SequenceOutput>> parent_child_dict;
      for (auto& sample : samples) {
        parent_child_dict[sample.parent_seq_id].push_back(sample);
      }
      std::vector<std::tuple<Sequence, Sequence>> child_seqs;

      for (auto& parent : parent_seqs) {
        auto& child_samples = parent_child_dict[parent.seq_id];

        if (child_samples.empty()) {
          seq_group.seqs_dict.at(parent.seq_id).status = SequenceStatus::kFinishedAborted;
          seq_group.seqs_dict.erase(parent.seq_id);
          queued_seq_group.seqs_dict.at(parent.seq_id).status = SequenceStatus::kFinishedAborted;
          queued_seq_group.seqs_dict.erase(parent.seq_id);
          scheduler_->FreeSeq(parent);
          continue;
        }

        if (child_samples.size() > 1) {
          for (int j = 0; j < child_samples.size() - 1; j++) {
            int new_child_seq_id = seq_count_++;
            auto& child_sample = child_samples[j];
            auto child_seq = parent;
            child_seq.seq_id = new_child_seq_id;
            child_seqs.emplace_back(child_seq, parent);
          }
        }

        auto& last_child_sample = child_samples.back();
        seq_group.seqs_dict.at(parent.seq_id).AppendTokenId(last_child_sample.output_token);
        queued_seq_group.seqs_dict.at(parent.seq_id).AppendTokenId(last_child_sample.output_token);
        child_seqs.emplace_back(parent, parent);
      }

      for (auto& [seq, parent_seq] : child_seqs) {
        // generate the output text
        std::vector<int> all_input_ids;
        all_input_ids.reserve(seq.data.GetLen());
        all_input_ids.insert(all_input_ids.end(),
                             seq.inputs.prompt_tokens_ids.begin(),
                             seq.inputs.prompt_tokens_ids.end());
        all_input_ids.insert(all_input_ids.end(),
                             seq.data.output_token_ids.begin(),
                             seq.data.output_token_ids.end());
        int token_generated = all_input_ids.back();
        std::span all_ids_span{all_input_ids.data(), all_input_ids.size()};
        size_t start = std::max(all_input_ids.size() - 5, size_t{0});
        std::string prefix_text = tokenizer_->Decode(
            all_ids_span.subspan(start, all_ids_span.size() - 1 - start));
        std::string full_text = tokenizer_->Decode(
            all_ids_span.subspan(start, all_ids_span.size() - start));

        std::string new_text = full_text.substr(prefix_text.size());

        seq.output_text = seq.output_text + new_text;
        queued_seq_group.seqs_dict.at(seq.seq_id).output_text = seq.output_text;
        std::cout << "seq.output_text = " << seq.output_text << std::endl;

        // stop check
        size_t new_char_count = new_text.size();
        auto& sampling_params = seq_group.sampling_params;

        if (seq.data.output_token_ids.size() < sampling_params.min_tokens)
          continue;

        if (!sampling_params.ignore_eos &&
            token_generated == model_->config_->model.eos_token_id) {
          seq.status = SequenceStatus::kFinishedStopped;
          seq_group.seqs_dict.at(seq.seq_id).status = SequenceStatus::kFinishedStopped;
          queued_seq_group.seqs_dict.at(seq.seq_id).status = SequenceStatus::kFinishedStopped;
        }

        if (seq.GetLen() > model_->config_->model.context_length) {
          seq.status = SequenceStatus::kFinishedLengthCapped;
          seq_group.seqs_dict.at(seq.seq_id).status = SequenceStatus::kFinishedLengthCapped;
          queued_seq_group.seqs_dict.at(seq.seq_id).status = SequenceStatus::kFinishedLengthCapped;
        }

        if (sampling_params.max_tokens > 0 &&
            seq.data.output_token_ids.size() >= sampling_params.max_tokens) {
          seq.status = SequenceStatus::kFinishedLengthCapped;
          seq_group.seqs_dict.at(seq.seq_id).status = SequenceStatus::kFinishedLengthCapped;
          queued_seq_group.seqs_dict.at(seq.seq_id).status = SequenceStatus::kFinishedLengthCapped;
        }
      }

      for (auto& [seq, parent_seq] : child_seqs) {
        if (seq.seq_id != parent_seq.seq_id) {
          seq_group.Add(seq);
          queued_seq_group.Add(seq);
          if (!seq.IsFinished()) {
            scheduler_->ForkSeq(parent_seq, seq);
          }
        }
      }

      for (auto& [seq, parent_seq] : child_seqs) {
        if (seq.seq_id == parent_seq.seq_id && seq.IsFinished()) {
          scheduler_->FreeSeq(seq);
        }
      }
    }
  }
  scheduler_->FreeFinishedSeqGroups();

  // create output
  std::vector<RequestOutput> request_outputs;
  for (auto& scheduled_seq_group : scheduled_seq_groups) {
    SequenceGroup& seq_group = scheduled_seq_group.seq_group;
    seq_group.MaybeSetFirstTokenTime(now);

    auto seqs = seq_group.GetSeqs();
    std::vector<Sequence> topn_seqs;
    if (seqs.size() == 1) {
      topn_seqs = seqs;
    } else {
      topn_seqs = std::vector<Sequence>(
          seqs.begin(), seqs.begin() + std::min(seq_group.sampling_params.n,
                                                static_cast<int>(seqs.size())));
    }
    std::vector<CompletionOutput> outs;
    for (auto& seq : topn_seqs) {
      outs.emplace_back(CompletionOutput{seq.seq_id, seq.output_text,
                                         seq.data.output_token_ids, 0, 0, GetFinishReason(seq.status), ""});
    }
    if (seq_group.IsFinished()) {
      seq_group.metrics.finished_time = now;
    }
    request_outputs.push_back(RequestOutput{
        seq_group.request_id, seq_group.GetSeqs().at(0).inputs.prompt,
        seq_group.GetSeqs().at(0).inputs.prompt_tokens_ids, outs,
        seq_group.IsFinished()});
  }
  return request_outputs;
}

}  // namespace Generators