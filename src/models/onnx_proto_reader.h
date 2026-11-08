// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <istream>
#include <string>

namespace Generators {

// Walks the fields of one protobuf message that occupies the bytes [begin, end) of a stream. Only the headers of the fields
// are read: a length-delimited payload is skipped unless ReadPayload() asks for it, so the initializers that make up the bulk
// of a model file are never touched. A malformed message (a truncated field, a length that leaves the message, a group) ends
// the walk with Failed() set.
class ProtoFieldReader {
 public:
  struct Field {
    uint32_t number{};
    bool length_delimited{};
    uint64_t payload_begin{};
    uint64_t payload_size{};
    uint64_t varint_value{};
  };

  ProtoFieldReader(std::istream& in, uint64_t begin, uint64_t end) : in_(in), pos_(begin), end_(end) {}

  bool Failed() const { return failed_; }

  // Moves to the next field; false at the end of the message and when it is malformed.
  bool Next(Field& field) {
    if (failed_ || pos_ >= end_) return false;
    uint64_t tag = 0;
    if (!ReadVarint(tag)) return Fail();
    const uint64_t number = tag >> 3;
    if (number == 0 || number > kMaxFieldNumber) return Fail();
    field = Field{static_cast<uint32_t>(number), false, 0, 0, 0};
    switch (tag & 7u) {
      case 0: {  // varint
        return ReadVarint(field.varint_value) || Fail();
      }
      case 1:  // 64-bit
        return Skip(8);
      case 2: {  // length-delimited
        uint64_t size = 0;
        if (!ReadVarint(size) || size > end_ - pos_) return Fail();
        field.length_delimited = true;
        field.payload_begin = pos_;
        field.payload_size = size;
        return Skip(size);
      }
      case 5:  // 32-bit
        return Skip(4);
      default:  // groups and unknown wire types
        return Fail();
    }
  }

  // The payload of a length-delimited field that Next() returned.
  bool ReadPayload(const Field& field, std::string& bytes) {
    bytes.resize(static_cast<size_t>(field.payload_size));
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(field.payload_begin));
    in_.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream_at_pos_ = false;
    return static_cast<bool>(in_) || Fail();
  }

 private:
  bool Fail() {
    failed_ = true;
    return false;
  }

  bool Skip(uint64_t count) {
    if (count > end_ - pos_) return Fail();
    pos_ += count;
    stream_at_pos_ = false;
    return true;
  }

  bool ReadVarint(uint64_t& value) {
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 7) {
      if (pos_ >= end_) return false;
      if (!stream_at_pos_) {
        in_.clear();
        in_.seekg(static_cast<std::streamoff>(pos_));
        stream_at_pos_ = true;
      }
      const std::istream::int_type byte = in_.get();
      if (byte == std::istream::traits_type::eof()) return false;
      ++pos_;
      value |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) return true;
    }
    return false;
  }

  std::istream& in_;
  uint64_t pos_;
  uint64_t end_;
  bool stream_at_pos_{false};  // whether the stream is positioned at pos_
  bool failed_{false};

  static constexpr uint64_t kMaxFieldNumber = (uint64_t{1} << 29) - 1;  // the protobuf limit
};

}  // namespace Generators
