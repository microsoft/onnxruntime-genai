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
 * [params setInputSequences:sequences error:&error];
 * [params setSearchOption:@"max_length" doubleValue:200 error:&error];
 *
 * OGASequences* output_sequences = [model generate:params error:&error];
 * NSString* out_string = [tokenizer decode:[output_sequences sequenceDataAtIndex:0] length:[output_sequences sequenceCountAtIndex:0] error:&error];
 *
 */

NS_ASSUME_NONNULL_BEGIN

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
 * @param tokensData The sequences data to be encoded.
 * @param tokensLength The length of the sequences data to be encoded.
 * @param error Optional error information set if an error occurs.
 * @return The decoding result, or nil if an error occurs.
 */
- (nullable NSString*)decode:(const int32_t*)tokensData
                      length:(size_t)tokensLength
                       error:(NSError**)error;

@end

/**
 * A tokenizer stream is used to decode token strings incrementally, one token at a time.
 */
@interface OGATokenizerStream : NSObject
- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a tokenizer stream with an underlying tokenizer.
 *
 * @param tokenizer The underlying tokenizer to use.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithTokenizer:(OGATokenizer*)tokenizer
                                     error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Creates a tokenizer stream with a multi modal processor
 *
 * @param processor The underlying processor to use.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithMultiModalProcessor:(OGAMultiModalProcessor*)processor
                                               error:(NSError**)error NS_DESIGNATED_INITIALIZER;
/**
 * Decode one token.
 *
 * @param token The token to be decoded.
 * @param error Optional error information set if an error occurs.
 * @return The decoding result, or nil if an error occurs.
 */
- (nullable NSString*)decode:(int32_t)token
                       error:(NSError**)error;
@end

/**
 * A series of generated sequences
 */
@interface OGASequences : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * The count of generated sequences
 * @param error Optional error information set if an error occurs.
 * @return The count of sequences, or size_t(-1) if an error occurs.
 */
- (size_t)getCountWithError:(NSError**)error NS_SWIFT_NAME(count());

/**
 * Retrieve the sequence data at the given index.
 * @param index The index needed.
 * @param error Optional error information set if an error occurs.
 * @return The sequence data at the given index, or nil if an error occurs.
 */
- (nullable const int32_t*)sequenceDataAtIndex:(size_t)index
                                         error:(NSError**)error;

/**
 * Retrieve the sequence count at the given index.
 * @param index The index needed.
 * @param error Optional error information set if an error occurs.
 * @return The sequence count at the given index, or size_t(-1) if an error occurs.
 */
- (size_t)sequenceCountAtIndex:(size_t)index
                         error:(NSError**)error;

@end

/**
 * The parameters for generation.
 */
@interface OGAGeneratorParams : NSObject
- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates a GeneratorParams from the given model.
 * @param model The model to use for generation.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithModel:(OGAModel*)model
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;

/**
 * Set input with NamedTensors type.
 * @param namedTensors The named tensors.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)setInputs:(OGANamedTensors*)namedTensors
            error:(NSError**)error;

/**
 * Set input with name and corresponding tensor.
 * @param name The input name.
 * @param tensor The tensor.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)setModelInput:(NSString*)name
               tensor:(OGATensor*)tensor
                error:(NSError**)error;

/**
 * Set double option value.
 * @param key The option key.
 * @param value The option value.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)setSearchOption:(NSString*)key
            doubleValue:(double)value
                  error:(NSError**)error;
/**
 * Set boolean option value.
 * @param key The option key.
 * @param value The option value.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)setSearchOption:(NSString*)key
              boolValue:(BOOL)value
                  error:(NSError**)error;
@end

/**
 * The main generator interface that can be used for generation loop.
 */
@interface OGAGenerator : NSObject

- (instancetype)init NS_UNAVAILABLE;
/**
 * Creates a generator.
 *
 * @param model The model to use.
 * @param params The generation params to use.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithModel:(OGAModel*)model
                                params:(OGAGeneratorParams*)params
                                 error:(NSError**)error NS_DESIGNATED_INITIALIZER;
/**
 * Whether generation is done.
 * @param error Optional error information set if an error occurs.
 * @return The result, or false if an error occurs.
 */
- (BOOL)isDoneWithError:(NSError**)error __attribute__((swift_error(nonnull_error)));

/**
 * Appends token sequences to the generator.
 * @param sequences The sequences to append.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)appendTokenSequences:(OGASequences*)sequences error:(NSError**)error;

/**
 * Appends token sequences to the generator.
 * @param tokens The tokens to append.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)appendTokens:(NSArray<NSNumber*>*)tokens error:(NSError**)error;

/**
 * Rewinds the generator to the given length.
 * @param newLength The desired length in tokens after rewinding.
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)rewindTo:(size_t)newLength error:(NSError**)error;

/**
 * Generate next token
 * @param error Optional error information set if an error occurs.
 */
- (BOOL)generateNextTokenWithError:(NSError**)error;
/**
 * Get the output tensor.
 * @param name The output name.
 * @param error Optional error information set if an error occurs.
 * @return The result, or nil if an error occurs.
 */
- (nullable OGATensor*)getOutput:(NSString*)name error:(NSError**)error;

/**
 * Retrieve the sequence data at the given index.
 * @param index The index needed.
 * @param error Optional error information set if an error occurs.
 * @return The sequence data at the given index, or nil if an error occurs.
 */
- (nullable const int32_t*)sequenceDataAtIndex:(size_t)index
                                         error:(NSError**)error;

/**
 * Retrieve the sequence count at the given index.
 * @param index The index needed.
 * @param error Optional error information set if an error occurs.
 * @return The sequence count at the given index, or size_t(-1) if an error occurs.
 */
- (size_t)sequenceCountAtIndex:(size_t)index
                         error:(NSError**)error;

/**
 * Clean up the resource before process exits.
 */
+ (void)shutdown;

@end

@interface OGATensor : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable instancetype)initWithDataPointer:(void*)data
                                       shape:(NSArray<NSNumber*>*)shape
                                        type:(OGAElementType)elementType
                                       error:(NSError**)error;
- (OGAElementType)getTypeWithError:(NSError**)error NS_SWIFT_NAME(type());
- (nullable void*)getDataPointerWithError:(NSError**)error NS_SWIFT_NAME(dataPointer());

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

- (nullable NSString*)decode:(const int32_t*)tokensData
                      length:(size_t)tokensLength
                       error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END
