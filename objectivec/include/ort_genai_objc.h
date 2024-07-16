// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

NS_ASSUME_NONNULL_BEGIN

@class OGASequences;
@class OGAGeneratorParams;

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

- (nullable NSString *)decode:(NSData *)data
                        error:(NSError **)error;
@end

@interface OGASequences : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithError:(NSError **)error NS_DESIGNATED_INITIALIZER;

- (size_t)count;
- (nullable NSData *)sequenceAtIndex:(size_t)index;

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

NS_ASSUME_NONNULL_END
