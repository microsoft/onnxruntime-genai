// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Master include file for all OGA wrapper classes

#include "oga_object.h"
#include "oga_borrowed_view.h"  // Define BorrowedArrayView template first
#include "oga_utils.h"  // Then utils which uses the type aliases
#include "oga_config.h"
#include "oga_model.h"
#include "oga_tokenizer.h"
#include "oga_generator.h"
#include "oga_tensor.h"
#include "oga_multimodal.h"
#include "oga_adapters.h"
#include "oga_engine.h"
