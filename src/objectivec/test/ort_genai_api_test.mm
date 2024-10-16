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

    OGATensor* tensor = [[OGATensor alloc] initWithDataPointer:data.data() shape:shape.data() type:OGAElementTypeFloat32 error:&error];

    [param setModelInput:@"test_input" tensor:tensor error:&error];
}

@end

NS_ASSUME_NONNULL_END