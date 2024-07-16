
#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGATokenizer {
    std::unique_ptr<OgaTokenizer> _tokenizer;
}


- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _tokenizer = OgaTokenizer::Create([model CXXAPIOgaModel]);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGASequences *)encode:(NSString *)str
                            error:(NSError **)error {
    OGASequences *sequences = [[OGASequences alloc] initWithError:error];
    if (error) {
        return nil;
    }
    _tokenizer->Encode([str UTF8String], [sequences CXXAPIOgaSequences]);
    return sequences;
}

- (nullable NSString *)decode:(NSData *)data
                        error:(NSError **)error {
    try {
        OgaString result = _tokenizer->Decode((const int32_t *)[data bytes], [data length]);
        return [NSString stringWithUTF8String:result];
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaTokenizer&)CXXAPIOgaTokenizer {
    return *(_tokenizer.get());
}

@end
