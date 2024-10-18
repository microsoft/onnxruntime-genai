
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_genai_objc.h"
#import <vector>
#import <array>

NS_ASSUME_NONNULL_BEGIN

@interface ORTGenAIModelTest : XCTestCase

@end

@implementation ORTGenAIModelTest


- (void)GreedySearchGptFp32 {
    std::vector<int64_t> input_ids_shape{2, 4};
    std::vector<int32_t> input_ids{0, 0, 0, 52, 0, 0, 195, 731};

    std::vector<int32_t> expected_output{
        0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
        0, 0, 195, 731, 731, 114, 114, 114, 114, 114};

    const auto max_length = 10;
    const auto batch_size = input_ids_shape[0];
    const auto input_sequence_length = input_ids_shape[1];

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* path = [[bundle resourcePath] stringByAppendingString:@"hf-internal-testing/tiny-random-gpt2-fp32"];

    NSError *error = nil;
    OGAModel* model = [[OGAModel alloc] initWithPath:path error:&error];

    OGAGeneratorParams *params = [[OGAGeneratorParams alloc] initWithModel:model error:&error];
    [params setSearchOption:@"max_length", max_length];
    [params setSearchOption:@"do_sample", YES];
    [params setSearchOption:@"top_p", 0.25];

    [params setInputIds:input_ids.data()
          inputIdsCount:input_ids.size()
         sequenceLength:input_sequence_length
              batchSize:batch_size
                   error:&error];
    OGAGenerator* generator = [[OGAGenerator alloc] initWithModel:model
                                                           params:params
                                                            error:&error];

    while (![generator isDone]) {
        [generator computeLogits];
        [generator generateNextToken];
    }

    for (int i = 0; i < batch_size; i++) {
        OGASequence *sequence = [generator sequenceAtIndex: i];
        auto* expected_output_start = &expected_output[i * max_length];
        XCTAssertTrue(0 == std::memcmp(expected_output_start, [sequence data], max_length * sizeof(int32_t)));
    }
}

@end