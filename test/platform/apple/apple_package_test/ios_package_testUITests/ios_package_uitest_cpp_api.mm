// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
//  ios_package_test_cpp_api.mm
//  ios_package_test_cpp_api
//
//  This file hosts the tests of ORT GenAI C++ API
//

#import <XCTest/XCTest.h>
#include <math.h>
#include <ort_genai.h>


@interface ios_package_test_cpp_api : XCTestCase

@end

@implementation ios_package_test_cpp_api

- (void)setUp {
  // Put setup code here. This method is called before the invocation of each test method in the class.

  // In UI tests it is usually best to stop immediately when a failure occurs.
  self.continueAfterFailure = YES;

  // In UI tests it’s important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
}

- (void)tearDown {
  // Put teardown code here. This method is called after the invocation of each test method in the class.
    OgaShutdown();
}

- (NSString*)getFilePath {
    NSBundle* bundle = [NSBundle bundleForClass:[self class]];
    NSString* path = [bundle resourcePath];
    return path;
}

- (void)testCppAPI_Basic {
    auto model = OgaModel::Create([self getFilePath].UTF8String);

    auto tokenizer = OgaTokenizer::Create(*model);

    const char* prompt = "<|system|>You are a helpful AI assistant.<|end|><|user|>Can you introduce yourself?<|end|><|assistant|>";

    auto sequences = OgaSequences::Create();
    tokenizer->Encode(prompt, *sequences);

    auto params = OgaGeneratorParams::Create(*model);
    params->SetSearchOption("max_length", 100);
    params->SetSearchOption("batch_size", 1);

    auto generator = OgaGenerator::Create(*model, *params);
    generator->AppendTokenSequences(*sequences);

    while (!generator->IsDone()) {
      generator->GenerateNextToken();
    }
    const auto output_sequence_length = generator->GetSequenceCount(0);
    const auto* output_sequence_data = generator->GetSequenceData(0);
    auto out_string = tokenizer->Decode(output_sequence_data, output_sequence_length);
}

@end
