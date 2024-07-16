// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#import "cxx_api.h"


NS_ASSUME_NONNULL_BEGIN

@interface OGAModel ()

- (const OgaModel&)CXXAPIOgaModel;

@end

@interface OGATokenizer ()

- (const OgaTokenizer&)CXXAPIOgaTokenizer;

@end

@interface OGASequences ()

- (instancetype)initWithNativeSeqquences:(std::unique_ptr<OgaSequences>)ptr;

- (OgaSequences&)CXXAPIOgaSequences;

@end

@interface OGAGeneratorParams ()

- (OgaGeneratorParams&)CXXAPIOgaGeneratorParams;

@end

NS_ASSUME_NONNULL_END
