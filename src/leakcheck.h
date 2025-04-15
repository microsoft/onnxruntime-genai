// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file will track the number of instances of each type that are created and destroyed. This is useful for
// debugging memory leaks. To use this, just add the type to the LeakTypeList in this file. Then have that type
// inherit from LeakChecked<(itself)>.
//
// On process exit, ValidateShutdown() will call LeakTypeList::Dump() and print out any types that have leaked.

namespace Generators {
struct Engine;
struct GeneratorParams;
struct Generator;
struct Model;
struct Request;
struct Search;
struct Tensor;
struct Tokenizer;
struct TokenizerStream;

template <typename... Types>
struct LeakTypeList {
  template <typename T>
  static constexpr bool is_tracked = (std::is_same_v<T, Types> || ...);
  static bool Dump();
};

using LeakTypes = LeakTypeList<Engine, GeneratorParams, Generator, Model, Request, Search, Tensor, Tokenizer, TokenizerStream>;

template <typename T>
struct LeakChecked {
  static_assert(LeakTypes::is_tracked<T>, "Please add this type to 'TrackedTypes' above");

  LeakChecked() { ++count_; }
  ~LeakChecked() { --count_; }

  static int Count() { return count_; }

 private:
  static inline std::atomic<int> count_;
};

}  // namespace Generators