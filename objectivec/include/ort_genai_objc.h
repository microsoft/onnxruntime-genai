NS_ASSUME_NONNULL_BEGIN

@interface OGAModel : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (nullable)initWithConfigPath:(NSString *)path
                         error:(NSError **)error NS_DESIGNATED_INITIALIZER;


@end

NS_ASSUME_NONNULL_END
