// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GenAI Objective-C API
//
// This is a wrapper around the C++ API, and provides for a set of Objective-C classes with automatic resource management

/* A simple end to end example of how to generate an answer from a prompt:
 *
 * OGAModel* model = [[OGAModel alloc] initWithPath:path error:&error];
 * OGATokenizer* tokenizer = [[OGATokenizer alloc] initWithModel:model error:&error];
 *
 * OGASequences* sequences = [tokenizer encode:@"A great recipe for Kung Pao chicken is " error:&error];
 *
 * OGAGeneratorParams* params = [[OGAGeneratorParams alloc] initWithModel:model error:&error];
 * [params setInputSequences:sequences];
 * [params setSearchOption:@"max_length" doubleValue:200 error:&error];
 *
 * OGASequences* output_sequences = [model generate:params error:&error];
 * NSString* out_string = [tokenizer decode:[output_sequences sequenceAtIndex:0]];
 *
 */

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


/**
 * An ORT GenAI model.
 */
@interface OGAModel : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a model.
 *
 * @param path The path to the ONNX GenAI model folder.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithPath:(NSString*)path
                                error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Generate sequences with the model.
 * The inputs and outputs are pre-allocated.
 *
 * @param params The generation params to use.
 * @param error Optional error information set if an error occurs.
 * @return The generated sequences.
 */
- (nullable OGASequences*)generate:(OGAGeneratorParams*)params
                             error:(NSError**)error;

@end

/**
 * An ORT GenAI tokenizer.
 */
@interface OGATokenizer : NSObject
- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a tokenizer.
 *
 * @param model The model to use.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithModel:(OGAModel*)model
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Encode text to sequences
 *
 * @param str The text to be encoded.
 * @param error Optional error information set if an error occurs.
 * @return The encoding result, or nil if an error occurs.
 */
- (nullable OGASequences*)encode:(NSString*)str
                           error:(NSError**)error;

/**
 * Decode sequences to text
 *
 * @param data The sequences data to be encoded.
 * @param error Optional error information set if an error occurs.
 * @return The decoding result, or nil if an error occurs.
 */
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
- (size_t)size;
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
