// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <XCTest/XCTest.h>

#import "ort_genai_objc.h"
#import "assertion_utils.h"
#import <vector>
#import <array>

NS_ASSUME_NONNULL_BEGIN

@interface ORTGenAIAPITest : XCTestCase

@end

@implementation ORTGenAIAPITest

- (void)setUp {
    [super setUp];
    self.continueAfterFailure = NO;
}

+ (void)tearDown {
    [OGAGenerator shutdown];
}

+ (NSString*)getModelPath {
    NSBundle* bundle = [NSBundle bundleForClass:[ORTGenAIAPITest class]];
    NSString* path = [[bundle resourcePath] stringByAppendingString:@"/tiny-random-LlamaForCausalLM-fp32"];
    return path;
}

- (void)testTensor_And_AddExtraInput {
    // Create a [3 4] shaped tensor
    std::array<float, 12> data{0, 1, 2, 3,
                             10, 11, 12, 13,
                             20, 21, 22, 23};
    NSArray<NSNumber*>* shape = @[@3, @4];

    NSError *error = nil;
    BOOL ret = NO;
    OGAModel* model = [[OGAModel alloc] initWithPath:[ORTGenAIAPITest getModelPath] error:&error];
    ORTAssertNullableResultSuccessful(model, error);

    OGAGeneratorParams *params = [[OGAGeneratorParams alloc] initWithModel:model error:&error];
    ORTAssertNullableResultSuccessful(params, error);

    OGATensor* tensor = [[OGATensor alloc] initWithDataPointer:data.data() shape:shape type:OGAElementTypeFloat32 error:&error];
    ORTAssertNullableResultSuccessful(tensor, error);

    ret = [params setModelInput:@"test_input" tensor:tensor error:&error];
    ORTAssertBoolResultSuccessful(ret, error);
}

- (void)testGetOutput {
    std::vector<int64_t> input_ids_shape{2, 4};
    NSArray<NSNumber*>* input_ids = @[@0, @0, @0, @52, @0, @0, @195, @731];
    const auto batch_size = input_ids_shape[0];
    int max_length = 10;

    NSError *error = nil;
    BOOL ret = NO;
    OGAModel* model = [[OGAModel alloc] initWithPath:[ORTGenAIAPITest getModelPath] error:&error];
    ORTAssertNullableResultSuccessful(model, error);

    OGAGeneratorParams *params = [[OGAGeneratorParams alloc] initWithModel:model error:&error];
    ORTAssertNullableResultSuccessful(params, error);

    [params setSearchOption:@"max_length" doubleValue:max_length error:&error];
    XCTAssertNil(error);

    [params setSearchOption:@"batch_size" doubleValue:batch_size error:&error];
    XCTAssertNil(error);

    OGAGenerator* generator = [[OGAGenerator alloc] initWithModel:model
                                                           params:params
                                                            error:&error];
    ORTAssertNullableResultSuccessful(generator, error);
    [generator appendTokens:input_ids error:&error];
    XCTAssertNil(error);

    // check prompt
    // full logits has shape [2, 4, vocab_size]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 4, 5]
    std::vector<float> expected_sampled_logits_prompt{-0.0682238f, 0.0405136f, 0.057766f, -0.0431961f, 0.00696388f,
                                                       -0.0153187f, 0.0369705f, 0.0259072f, -0.0189864f, 0.010939f,
                                                       -0.007559f, 0.0976457f, -0.0195211f, -0.0496172f, -0.0826776f,
                                                       -0.061368f, 0.0905409f, 0.0395047f, 0.0156607f, -0.124637f,
                                                       0.0302449f, 0.0105196f, -0.0475081f, 0.18416f, -0.102302f,
                                                       0.0363197f, -0.0178498f, 0.0538303f, -0.15488f, 0.0186949f,
                                                       -0.308369f, -0.150942f, 0.0628686f, 0.121276f, -0.043074f,
                                                       0.0784324f, -0.0752792f, 0.0352388f, -0.0203399f, -0.0446295f};

    OGATensor* prompt_logits_ptr = [generator getOutput:@"logits" error:&error];
    ORTAssertNullableResultSuccessful(prompt_logits_ptr, error);
    auto prompt_logits = static_cast<float*>([prompt_logits_ptr getDataPointerWithError:&error]);
    XCTAssertNil(error);
    XCTAssertNotEqual(prompt_logits, nullptr);
    const int num_prompt_outputs_to_check = 40;
    const int sample_size = 200;
    const float tolerance = 0.001f;
    // Verify outputs match expected outputs
    for (int i = 0; i < num_prompt_outputs_to_check; i++) {
        XCTAssertEqualWithAccuracy(expected_sampled_logits_prompt[i], prompt_logits[i * sample_size], tolerance);
    }

    ret = [generator generateNextTokenWithError:&error];
    ORTAssertBoolResultSuccessful(ret, error);
    ret = [generator generateNextTokenWithError:&error];
    ORTAssertBoolResultSuccessful(ret, error);

    // check for the 1st token generation
    // full logits has shape [2, 1, 1000]. Sample 1 for every 200 tokens and the expected sampled logits has shape [2, 1, 5]
    std::vector<float> expected_sampled_logits_token_gen{-0.0966602f, 0.0653766f, -0.0240025f, -0.238864f, 0.0626191f,
                                                          0.0217852f, 0.0282981f, 0.0627022f, -0.0670064f, -0.0286431f};

    OGATensor* token_gen_logits_ptr = [generator getOutput:@"logits" error:&error];
    ORTAssertNullableResultSuccessful(token_gen_logits_ptr, error);

    auto token_gen_logits = static_cast<float*>([token_gen_logits_ptr getDataPointerWithError:&error]);
    XCTAssertNil(error);
    XCTAssertNotEqual(token_gen_logits, nullptr);
    int num_token_gen_outputs_to_check = 10;

    for (int i = 0; i < num_token_gen_outputs_to_check; i++) {
        XCTAssertEqualWithAccuracy(expected_sampled_logits_token_gen[i], token_gen_logits[i * sample_size], tolerance);
    }
    [generator generateNextTokenWithError:&error];
    ORTAssertBoolResultSuccessful(ret, error);
}

@end

NS_ASSUME_NONNULL_END
