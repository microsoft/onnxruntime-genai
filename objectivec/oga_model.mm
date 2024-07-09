#import "ort_genai_objc.h"
#import "cxx_api.h"
#import "error_utils.h"

@implementation OGAModel {
    std::unique_ptr<OgaModel> _model;
}


- (nullable)initWithConfigPath:(NSString *)path
                         error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }
    
    try {
        _model = OgaModel::Create("phi-2");
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
    
}


@end
