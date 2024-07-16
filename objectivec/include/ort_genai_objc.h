// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

NS_ASSUME_NONNULL_BEGIN


@class OGASpan;
@class OGASequences;
@class OGAGeneratorParams;
@class OGATokenizerStream;


@interface OGAModel : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithConfigPath:(NSString *)path
                         error:(NSError **)error NS_DESIGNATED_INITIALIZER;


- (nullable OGASequences *)generate:(OGAGeneratorParams *)params
                              error:(NSError **)error;

@end


@interface OGATokenizer : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (nullable OGASequences *)encode:(NSString *)str
                            error:(NSError **)error;

- (nullable NSString *)decode:(OGASpan *)data
                        error:(NSError **)error;

@end

@interface OGATokenizerStream: NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithTokenizer:(OGATokenizer *)tokenizer
                        error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (nullable NSString *)decode:(int32_t)token
                        error:(NSError **)error;
@end;


@interface OGASpan : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (int32_t)last;

@end

@interface OGASequences : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (size_t)count;
- (nullable OGASpan *)sequenceAtIndex:(size_t)index;

@end

@interface OGAGeneratorParams : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (BOOL)setInputSequences:(OGASequences *)sequences
                    error:(NSError **)error;

- (BOOL)setSearchOption:(NSString *)key
            doubleValue:(double)value
                  error:(NSError **)error;

- (BOOL)setSearchOption:(NSString *)key
              boolValue:(BOOL)value
                  error:(NSError **)error;
@end

@interface OGAGenerator : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithModel:(OGAModel *)model
                   params:(OGAGeneratorParams *)params
                    error:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (BOOL)isDone;
- (void)ComputeLogits;
- (void)GenerateNextToken;

- (nullable OGASpan *)sequenceAtIndex:(size_t) index;
@end

NS_ASSUME_NONNULL_END
