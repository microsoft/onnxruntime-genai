// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

#include <stdexcept>

@implementation OGAGenerator {
  std::unique_ptr<OgaGenerator> _generator;
}

- (nullable instancetype)initWithModel:(OGAModel*)model
                                params:(OGAGeneratorParams*)params
                                 error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _generator = OgaGenerator::Create([model CXXAPIOgaModel], [params CXXAPIOgaGeneratorParams]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)isDoneWithError:(NSError**)error {
  try {
    return _generator->IsDone();
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setInputs:(OGANamedTensors*)namedTensors error:(NSError**)error {
  try {
    _generator->SetInputs([namedTensors CXXAPIOgaNamedTensors]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setModelInput:(NSString*)name tensor:(OGATensor*)tensor error:(NSError**)error {
  try {
    _generator->SetModelInput([name UTF8String], [tensor CXXAPIOgaTensor]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)appendTokenSequences:(OGASequences*)sequences error:(NSError**)error {
  try {
    _generator->AppendTokenSequences([sequences CXXAPIOgaSequences]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)appendTokens:(NSArray<NSNumber*>*)tokens error:(NSError**)error {
  try {
    std::vector<int32_t> cxxTokens;
    for (NSNumber* object in tokens) {
      cxxTokens.push_back([object intValue]);
    }

    _generator->AppendTokens(cxxTokens.data(), cxxTokens.size());
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (size_t)tokenCount:(NSError**)error {
  try {
    return _generator->TokenCount();
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_SIZE_T(error)
}

- (BOOL)rewindTo:(size_t)newLength error:(NSError**)error {
  try {
    _generator->RewindTo(newLength);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)generateNextTokenWithError:(NSError**)error {
  try {
    _generator->GenerateNextToken();
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (nullable OGATensor*)getInput:(NSString*)name error:(NSError**)error {
  try {
    std::unique_ptr<OgaTensor> input = _generator->GetInput([name UTF8String]);
    return [[OGATensor alloc] initWithCXXPointer:std::move(input)];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGATensor*)getOutput:(NSString*)name error:(NSError**)error {
  try {
    std::unique_ptr<OgaTensor> output = _generator->GetOutput([name UTF8String]);
    return [[OGATensor alloc] initWithCXXPointer:std::move(output)];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable const int32_t*)sequenceDataAtIndex:(size_t)index error:(NSError**)error {
  try {
    return _generator->GetSequenceData(index);
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (size_t)sequenceCountAtIndex:(size_t)index error:(NSError**)error {
  try {
    return _generator->GetSequenceCount(index);
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_SIZE_T(error)
}

- (BOOL)getSpeculativeStats:(OGASpeculativeStats*)stats error:(NSError**)error {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats must not be null.");
    }

    const auto source = _generator->GetSpeculativeStats();
    stats->rounds = source.rounds;
    stats->completedRounds = source.completed_rounds;
    stats->interruptedRounds = source.interrupted_rounds;
    stats->activeRounds = source.active_rounds;
    stats->draftTokensProposed = source.draft_tokens_proposed;
    stats->draftTokensEvaluated = source.draft_tokens_evaluated;
    stats->draftTokensAccepted = source.draft_tokens_accepted;
    stats->correctionTokens = source.correction_tokens;
    stats->bonusTokens = source.bonus_tokens;
    stats->tokensQueued = source.tokens_queued;
    stats->tokensEmitted = source.tokens_emitted;
    stats->tokensDiscarded = source.tokens_discarded;
    stats->tokensBuffered = source.tokens_buffered;
    stats->draftForwardPasses = source.draft_forward_passes;
    stats->targetForwardPasses = source.target_forward_passes;
    stats->formulaSupported = source.formula_supported != 0;
    stats->totalDraftMs = source.total_draft_ms;
    stats->totalTargetMs = source.total_target_ms;
    stats->totalReconciliationMs = source.total_reconciliation_ms;
    stats->avgDraftMsPerToken = source.avg_draft_ms_per_token;
    stats->acceptanceRate = source.acceptance_rate;
    stats->avgDraftTokensPerRound = source.avg_draft_tokens_per_round;
    stats->meanEmittedTokensPerRound = source.mean_emitted_tokens_per_round;
    stats->expectedTokensPerRound = source.expected_tokens_per_round;
    stats->avgTargetMsPerRound = source.avg_target_ms_per_round;
    stats->targetBaselineMsPerToken = source.target_baseline_ms_per_token;
    stats->targetOverheadRatio = source.target_overhead_ratio;
    stats->estimatedSpeedup = source.estimated_speedup;
    stats->observedSpeedup = source.observed_speedup;
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

+ (void)shutdown {
  OgaShutdown();
}
@end
