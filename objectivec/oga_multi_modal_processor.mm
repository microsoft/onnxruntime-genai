
#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OgaMultiModalProcessor {
    std::unique_ptr<OgaMultiModalProcessor> _processor;
}


- (nullable)initWithModel:(OGAModel *)model
                    error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _processor = OgaMultiModalProcessor::Create([model CXXAPIOgaModel]);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGANamedTensor *)processImages:(NSString *)prompt
                                  images:(OGAImages *)images
                                   error:(NSError **)error {
    try {
        OGANamedTensor *result = [OGANamedTensor alloc] initWithNativePointer:_processor->ProcessImages([prompt UTF8String], [images CXXAPIOgaImages])];
        return result;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSString *)decode:(OGASpan *) data
                        error:(NSError **)error {
    try {
        OgaString result = _processor->Decode(data.pointer, data.size);
        return [NSString stringWithUTF8String:result];
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaMultiModalProcessor&)CXXAPIOgaMultiModalProcessor {
    return *(_processor.get());
}
