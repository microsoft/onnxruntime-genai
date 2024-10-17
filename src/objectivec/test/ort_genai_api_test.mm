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
    auto input_sequence_length = input_ids_shape[1];
    auto batch_size = input_ids_shape[0];
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

}

@end

NS_ASSUME_NONNULL_END