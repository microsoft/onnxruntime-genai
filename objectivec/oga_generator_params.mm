#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGAGeneratorParams {
    std::unique_ptr<OgaGeneratorParams> _generatorParams;
}


- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _generatorParams = OgaGeneratorParams::Create([model CXXAPIOgaModel]);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)setInputSequences:(OGASequences *)sequences
                    error:(NSError **)error {
    try {
        _generatorParams->SetInputSequences([sequences CXXAPIOgaSequences]);
        return YES;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setSearchOption:(NSString *)key
            doubleValue:(double)value
                  error:(NSError **)error {
    try {
        _generatorParams->SetSearchOption([key UTF8String], value);
        return YES;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setSearchOption:(NSString *)key
              boolValue:(BOOL)value
                  error:(NSError **)error {
    try {
        _generatorParams->SetSearchOptionBool([key UTF8String], value);
        return YES;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}


- (OgaGeneratorParams&)CXXAPIOgaGeneratorParams {
    return *(_generatorParams.get());
}

@end
