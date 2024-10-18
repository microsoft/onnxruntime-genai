// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_genai_objc.h"
#import <vector>
#import <array>

NS_ASSUME_NONNULL_BEGIN

@interface ORTGenAIAPITest : XCTestCase

@end

@implementation ORTGenAIAPITest


- (void)Tensor_And_AddExtraInput {
    // Create a [3 4] shaped tensor
    std::array<float, 12> data{0, 1, 2, 3,
                             10, 11, 12, 13,
                             20, 21, 22, 23};
    std::vector<int64_t> shape{3, 4};  // Use vector so we can easily compare for equality later

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* path = [[bundle resourcePath] stringByAppendingString:@"hf-internal-testing/tiny-random-gpt2-fp32"];

    NSError *error = nil;
    OGAModel* model = [[OGAModel alloc] initWithPath:path error:&error];
    OGAGeneratorParams *param = [[OGAGeneratorParams alloc] initWithModel:model error:&error];

    OGAInt64Span* shapeData = [[OGAInt64Span alloc] initWithRawPointer:shape.data() size:2];
    OGATensor* tensor = [[OGATensor alloc] initWithDataPointer:data.data() shape:shapeData type:OGAElementTypeFloat32 error:&error];

    [param setModelInput:@"test_input" tensor:tensor error:&error];
}

- (void)GetOutput {
    std::vector<int64_t> input_ids_shape{2, 4};
    std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};
    const auto batch_size = input_ids_shape[0];
    const auto input_sequence_length = input_ids_shape[1];
    int max_length = 10;

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* path = [[bundle resourcePath] stringByAppendingString:@"hf-internal-testing/tiny-random-gpt2-fp32"];

    NSError *error = nil;
    OGAModel* model = [[OGAModel alloc] initWithPath:path error:&error];
    OGAGeneratorParams *params = [[OGAGeneratorParams alloc] initWithModel:model error:&error];

    [params setInputIds:input_ids.data()
          inputIdsCount:input_ids.size()
         sequenceLength:input_sequence_length
              batchSize:batch_size
                   error:&error];

    [params setSearchOption:@"max_length" doubleValue:max_length error:&error];

    OGAGenerator* generator = [[OGAGenerator alloc] initWithModel:model
                                                           params:params
                                                            error:&error];
    // check prompt
     // full logits has shape [2, 4, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 4, 5]
    std::vector<float> expected_sampled_logits_prompt{0.29694548f, 0.00955007f, 0.0430819f, 0.10063869f, 0.0437237f,
                                                    0.27329233f, 0.00841076f, -0.1060291f, 0.11328877f, 0.13369876f,
                                                    0.30323744f, 0.0545997f, 0.03894716f, 0.11702324f, 0.0410665f,
                                                    -0.12675379f, -0.04443946f, 0.14492269f, 0.03021223f, -0.03212897f,
                                                    0.29694548f, 0.00955007f, 0.0430819f, 0.10063869f, 0.0437237f,
                                                    0.27329233f, 0.00841076f, -0.1060291f, 0.11328877f, 0.13369876f,
                                                    -0.04699047f, 0.17915794f, 0.20838135f, 0.10888482f, -0.00277808f,
                                                    0.2938929f, -0.10538938f, -0.00226692f, 0.12050669f, -0.10622668f};

    [generator computeLogits];
    OGATensor* prompt_logits_ptr = [generator getOutput:@"logits"];
    auto prompt_logits = static_cast<float*>([prompt_logits_ptr data]);
    const int num_prompt_outputs_to_check = 40;
    const int sample_size = 200;
    const float tolerance = 0.001f;
    // Verify outputs match expected outputs
    for (int i = 0; i < num_prompt_outputs_to_check; i++) {
        XCTAssertEqualWithAccuracy(expected_sampled_logits_prompt[i], prompt_logits[i * sample_size], tolerance);
    }

    [generator generateNextToken];

    // check for the 1st token generation
    // full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
    std::vector<float> expected_sampled_logits_token_gen{0.03742531f, -0.05752287f, 0.14159015f, 0.04210977f, -0.1484456f,
                                                       0.3041716f, -0.08701379f, -0.03778192f, 0.07471392f, -0.02049096f};

    [generator computeLogits];
    OGATensor* token_gen_logits_ptr = [generator getOutput:@"logits"];
    auto token_gen_logits = static_cast<float*>([token_gen_logits_ptr data]);
    int num_token_gen_outputs_to_check = 10;

    for (int i = 0; i < num_token_gen_outputs_to_check; i++) {
        XCTAssertEqualWithAccuracy(expected_sampled_logits_token_gen[i], token_gen_logits[i * sample_size], tolerance);
    }
    [generator generateNextToken];
}

@end

NS_ASSUME_NONNULL_END