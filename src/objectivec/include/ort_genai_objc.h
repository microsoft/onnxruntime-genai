// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

NS_ASSUME_NONNULL_BEGIN

@class OGAInt32Span;
@class OGATensor;
@class OGASequences;
@class OGANamedTensors;
@class OGAGeneratorParams;
@class OGATokenizerStream;
@class OGAMultiModalProcessor;

typedef NS_ENUM(NSInteger, OGAElementType) {
  OGAElementTypeUndefined,
  OGAElementTypeFloat32,  // maps to c type float
  OGAElementTypeUint8,    // maps to c type uint8_t
  OGAElementTypeInt8,     // maps to c type int8_t
  OGAElementTypeUint16,   // maps to c type uint16_t
  OGAElementTypeInt16,    // maps to c type int16_t
  OGAElementTypeInt32,    // maps to c type int32_t
  OGAElementTypeInt64,    // maps to c type int64_t
  OGAElementTypeString,   // string type (not currently supported by Oga)
  OGAElementTypeBool,     // maps to c type bool
  OGAElementTypeFloat16,  // IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
  OGAElementTypeFloat64,  // maps to c type double
  OGAElementTypeUint32,   // maps to c type uint32_t
  OGAElementTypeUint64,   // maps to c type uint64_t
};

@interface OGAModel : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithPath:(NSString*)path
                                error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (nullable OGASequences*)generate:(OGAGeneratorParams*)params
                             error:(NSError**)error;

@end

@interface OGATokenizer : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithModel:(OGAModel*)model
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (nullable OGASequences*)encode:(NSString*)str
                           error:(NSError**)error;

- (nullable NSString*)decode:(OGAInt32Span*)data
                       error:(NSError**)error;

@end

@interface OGATokenizerStream : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithTokenizer:(OGATokenizer*)tokenizer
                                     error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (nullable instancetype)initWithMultiModalProcessor:(OGAMultiModalProcessor*)processor
                                               error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (nullable NSString*)decode:(int32_t)token
                       error:(NSError**)error;
@end

@interface OGAInt32Span : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithRawPointer:(const int32_t*)pointer size:(size_t)size;

- (const int32_t*)pointer;
- (size_t)size;
- (int32_t)last;

@end

@interface OGAInt64Span : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithRawPointer:(const int64_t*)pointer size:(size_t)size;

- (const int64_t*)pointer;
- (int64_t)last;

@end

@interface OGASequences : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (size_t)count;
- (nullable OGAInt32Span*)sequenceAtIndex:(size_t)index;

@end

@interface OGAGeneratorParams : NSObject
- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithModel:(OGAModel*)model
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (BOOL)setInputs:(OGANamedTensors*)namedTensors
            error:(NSError**)error;

- (BOOL)setInputIds:(const int32_t*)rawPointer
      inputIdsCount:(size_t)inputIdsCount
     sequenceLength:(size_t)sequenceLength
          batchSize:(size_t)batchSize
              error:(NSError**)error;

- (BOOL)setInputSequences:(OGASequences*)sequences
                    error:(NSError**)error;

- (BOOL)setModelInput:(NSString*)name
               tensor:(OGATensor*)tensor
                error:(NSError**)error;

- (BOOL)setSearchOption:(NSString*)key
            doubleValue:(double)value
                  error:(NSError**)error;

- (BOOL)setSearchOption:(NSString*)key
              boolValue:(BOOL)value
                  error:(NSError**)error;
@end

@interface OGAGenerator : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithModel:(OGAModel*)model
                                params:(OGAGeneratorParams*)params
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (BOOL)isDone;
- (void)computeLogits;
- (void)generateNextToken;
- (OGATensor*)getOutput:(NSString*)name;

- (nullable OGAInt32Span*)sequenceAtIndex:(size_t)index;

@end

@interface OGATensor : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithDataPointer:(void*)data
                                       shape:(OGAInt64Span*)shape
                                        type:(OGAElementType)elementType
                                       error:(NSError**)error;
- (OGAElementType)type;
- (void*)data;

@end

@interface OGANamedTensors : NSObject

- (instancetype)init NS_UNAVAILABLE;

@end

@interface OGAImages : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithPath:(NSArray<NSString*>*)paths
                                error:(NSError**)error NS_DESIGNATED_INITIALIZER;

@end

@interface OGAMultiModalProcessor : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithModel:(OGAModel*)model
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (nullable OGANamedTensors*)processImages:(NSString*)prompt
                                    images:(OGAImages*)images
                                     error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END
