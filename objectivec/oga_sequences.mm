#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGASequences {
    std::unique_ptr<OgaSequences> _sequences;
}

- (instancetype)initWithNativeSeqquences:(std::unique_ptr<OgaSequences>)ptr {
    _sequences = std::move(ptr);
    return self;
}

- (nullable)initWithError:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _sequences = OgaSequences::Create();
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (size_t)count {
    return _sequences->Count();
}

- (nullable NSData *)sequenceAtIndex:(size_t) index {
    if (index >= [self count]) {
        return nil;
    }
    size_t sequenceLength = _sequences->SequenceCount(index);
    const int32_t* data = _sequences->SequenceData(index);
    return [[NSData alloc] initWithBytes:data length:sequenceLength];
}

- (OgaSequences&) CXXAPIOgaSequences {
    return *(_sequences.get());
}

@end
