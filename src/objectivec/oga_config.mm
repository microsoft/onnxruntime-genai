// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGAConfig {
  std::unique_ptr<OgaConfig> _config;
}

- (nullable instancetype)initWithPath:(NSString*)path error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _config = OgaConfig::Create([path UTF8String]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)clearProvidersWithError:(NSError**)error {
  try {
    _config->ClearProviders();
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)appendProvider:(NSString*)provider error:(NSError**)error {
  try {
    _config->AppendProvider([provider UTF8String]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setProviderOption:(NSString*)provider key:(NSString*)key value:(NSString*)value error:(NSError**)error {  
  try {
    _config->SetProviderOption([provider UTF8String], [key UTF8String], [value UTF8String]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)overlay:(NSString*)json error:(NSError**)error {
  try {
    _config->Overlay([json UTF8String]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (const OgaConfig&)CXXAPIOgaConfig {
  return *(_config.get());
}

@end
