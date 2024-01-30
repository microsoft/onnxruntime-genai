// Licensed under the MIT License.
#include <gsl/narrow>

#include "token_bpe.h"

namespace tfm {

const char kModel_GPT2[] = "GPT2";
const char kModel_Roberta[] = "Roberta";
const char kModel_CLIP[] = "CLIP";
const char kModel_CodeGen[] = "CodeGen";

// Note: the following logic comes from CPython: unicodetype_db.h (_PyUnicode_IsWhitespace)
bool IsUnicodeSpace(char32_t ch) {
  const std::set<char32_t> unicode_spaces = {
      0x0009,  // CHARACTER TABULATION
      0x000A,  // LINE FEED (LF)
      0x000B,  // LINE TABULATION
      0x000C,  // FORM FEED (FF)
      0x000D,  // CARRIAGE RETURN (CR)
      0x001C,  // FILE SEPARATOR
      0x001D,  // GROUP SEPARATOR
      0x001E,  // RECORD SEPARATOR
      0x001F,  // UNIT SEPARATOR
      0x0020,  // SPACE
      0x0085,  // NEXT
      0x00A0,  // NO-BREAK SPACE
      0x1680,  // OGHAM SPACE MARK
      0x2000,  // EN QUAD
      0x2001,  // EM QUAD
      0x2002,  // EN SPACE
      0x2003,  // EM SPACE
      0x2004,  // THREE-PER-EM SPACE
      0x2005,  // FOUR-PER-EM SPACE
      0x2006,  // SIX-PER-EM SPACE
      0x2007,  // FIGURE SPACE
      0x2008,  // PUNCTUATION SPACE
      0x2009,  // THIN SPACE
      0x200A,  // HAIR SPACE
      0x2028,  // LINE SEPARATOR
      0x2029,  // PARAGRAPH SEPARATOR
      0x202F,  // NARROW NO-BREAK SPACE
      0x205F,  // MEDIUM MATHEMATICAL SPACE
      0x3000,  // IDEOGRAPHIC SPACE
  };
  return unicode_spaces.count(ch) > 0;
}

// only support latin now
char32_t ToLower(char32_t c) {
  if ((c >= 'A') && (c <= 'Z')) {
    return c + 'a' - 'A';
  } else if ((c >= U'À' && (c <= U'ß'))) {
    return c + U'à' - U'À';
  }
  return c;
}

bool AllSpaceUstring(const std::u32string& str) {
  return std::all_of(str.begin(), str.end(), [](char32_t ch) { return IsUnicodeSpace(ch); });
}

std::u32string RemoveConsecutiveSpaces(const std::u32string& input) {
  std::u32string result;
  result.reserve(input.size());
  bool lastWasSpace = false;

  for (auto ch : input) {
    if (IsUnicodeSpace(ch)) {
      if (!lastWasSpace) {
        result.push_back(ch);
      }
      lastWasSpace = true;
    } else {
      result.push_back(ch);
      lastWasSpace = false;
    }
  }

  return result;
}

void BPETokenizer::CreateByteEncoder() {
  char32_t index = 256;
  for (char32_t i = 0; i < 256; ++i) {
    /*
    bs = (
      list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    */
    if ((/* i >= 0 && */ i < 33) || (i >= 127 && i < 161) || (i == 173)) {
      byte_decoder_[index] = i;
      byte_encoder_[i] = bbpe_encoder_.GetTokenId(EncodeUTF8Char(index++));
    } else {
      byte_encoder_[i] = bbpe_encoder_.GetTokenId(EncodeUTF8Char(i));
      byte_decoder_[i] = i;
    }
  }
}

BPETokenizer::BPETokenizer() {
}

BPETokenizer::~BPETokenizer() = default;

std::string_view BPETokenizer::ModelName() const {
  return model_name_;
}

void BPETokenizer::LoadPredefinedTokens(const TokenConfig& config) {
  unk_token_ = config.unk_token_.content_;
  bos_token_ = config.bos_token_.content_;
  eos_token_ = config.eos_token_.content_;
  pad_token_ = config.pad_token_;

  unk_token_id_ = bbpe_encoder_.GetTokenId(unk_token_);
  bos_token_id_ = bbpe_encoder_.GetTokenId(bos_token_);
  eos_token_id_ = bbpe_encoder_.GetTokenId(eos_token_);
  pad_token_id_ = bbpe_encoder_.GetTokenId(pad_token_);

  added_tokens_.emplace(std::pair(unk_token_id_, unk_token_));
  added_tokens_.emplace(std::pair(bos_token_id_, bos_token_));
  added_tokens_.emplace(std::pair(eos_token_id_, eos_token_));
  added_tokens_.emplace(std::pair(pad_token_id_, pad_token_));

  all_special_ids_.emplace(unk_token_id_);
  all_special_ids_.emplace(bos_token_id_);
  all_special_ids_.emplace(eos_token_id_);
  all_special_ids_.emplace(pad_token_id_);
}

TfmStatus BPETokenizer::DecodeExtraArgs(const simdjson::dom::element& root) {
  simdjson::dom::element decoder_obj;
  auto error = root.at_key("decoder").get(decoder_obj);
  if (error != simdjson::SUCCESS && error != simdjson::NO_SUCH_FIELD) {
    return {kTfmErrorInvalidFile, "Cannot parse the decoder section in the the tokenizer.json"};
  }
  TryToGetJson(decoder_obj, "add_prefix_space", decode_extra_args_.add_prefix_space);
  return TfmStatus::OK();
}

TfmStatus BPETokenizer::Onload() {
  simdjson::dom::parser parser;
  simdjson::dom::element root;
  std::string tokenizer_file = GetDataDir() + "/tokenizer.json";
  auto error = parser.load(tokenizer_file).get(root);
  if (error) {
    return {kTfmErrorInvalidFile, "Failed to parse tokenizer.json"};
  }

  auto& config = *GetConfig();
  model_name_ = std::string_view(config.tokenizer_class_.c_str(),
                                 config.tokenizer_class_.find("Tokenizer"));
  auto status = bbpe_encoder_.Load(root, config);
  if (!status.ok()) {
    return status;
  }

  CreateByteEncoder();

  simdjson::dom::element added_tokens_obj;
  if (error = root["added_tokens"].get(added_tokens_obj); error) {
    // Get AddedTokens from config
    std::string_view added_tokens[] = {
        config.unk_token_.content_,
        config.eos_token_.content_,
        config.bos_token_.content_,
        config.pad_token_};
    size_t num_added_tokens = sizeof(added_tokens) / sizeof(added_tokens[0]);

    if (config.pad_token_.empty()) {
      num_added_tokens--;
    }
    if (config.bos_token_.content_.empty()) {
      num_added_tokens--;
    }

    status = extended_token_.LoadAddedTokens(added_tokens, num_added_tokens);
  } else {
    status = extended_token_.LoadAddedTokens(added_tokens_obj, added_tokens_);
  }

  if (!status.ok()) {
    return status;
  }

  LoadPredefinedTokens(config);
  arr_vocab_ = bbpe_encoder_.BuildDecoder();
  status = DecodeExtraArgs(root);

  return status;
}

std::vector<tfmTokenId_t> BPETokenizer::Encode(std::string_view sv_input,
                                               int64_t max_length,
                                               bool compute_offset_mapping,
                                               std::list<OffsetMappingType>& offset_map) const {
  std::vector<tfmTokenId_t> res;
  std::list<std::pair<uint32_t, uint32_t>> byte_list;

  std::u32string input = FromUTF8(sv_input);

  bool clean_up_spaces = false;
  if (ModelName() == kModel_CLIP) {
    clean_up_spaces = true;
    // Merges consecutive '\s+' for CLIP
    /*
      text = re.sub(r"\s+", " ", text)
      text = text.strip()
    */
    std::u32string str = RemoveConsecutiveSpaces(input);
    if (IsUnicodeSpace(str.front())) {
      str.erase(str.begin());
    }
    if (IsUnicodeSpace(str.back())) {
      str.pop_back();
    }
    // remove newlines as CLIP ignores them (treats them as whitespace which is then cleaned)
    str.erase(std::remove(str.begin(), str.end(), U'\n'), str.end());
    str.erase(std::remove(str.begin(), str.end(), U'\r'), str.end());
    input = str;
  }

  if (AllSpaceUstring(input) && ModelName() == kModel_CLIP) {
    // Add BOS and EOS token to result
    res.push_back(bos_token_id_);
    res.push_back(eos_token_id_);
    return res;
  }

  if (ModelName() != kModel_GPT2 && ModelName() != kModel_CodeGen) {
    // Add BOS token to result
    res.push_back(bos_token_id_);
  }
  if (ModelName() == kModel_CLIP) {
    // Convert to lowercase
    std::transform(input.begin(), input.end(), input.begin(), [](char32_t c) { return static_cast<char32_t>(ToLower(c)); });
  }

  // Parse input
  auto special_token_split_res = extended_token_.Split(input);
  bpe::TokenWithRegularExp regcmp;

  for (auto& seg_id : special_token_split_res) {
    if (static_cast<int64_t>(res.size()) >= max_length) break;

    if (seg_id.second != bpe::kInvalidTokenId) {
      res.push_back(seg_id.second);
      continue;
    }

    // Note: keep ptr to make sure the string_view is valid in the following process
    std::u32string str(seg_id.first);
    regcmp.Set(str);

    size_t offset = 0;
    OffsetMappingType offset_mapping;

    if (compute_offset_mapping) {
      if (ModelName() != kModel_GPT2) {
        // Add offset mapping for BOS token
        offset_mapping.emplace_back(0, 0);
      }
    }

    while (static_cast<int64_t>(res.size()) < max_length) {
      auto [b, tok] = regcmp.GetNextToken();

      if (!b) break;

      std::string utf8_token = ToUTF8(std::u32string(tok));

      size_t space_dif = 0;
      if (compute_offset_mapping) {
        // Handle special case for offset mapping
        if (utf8_token.at(0) == ' ') {
          offset++;
          space_dif = -1;  // account for spaces used in offset map algorithm in bpe(byte_list_)
        }
      }

      // Get byte encodings prior to performing BPE
      byte_list.clear();

      if (clean_up_spaces) {
        // Whitespace clean
        utf8_token.erase(std::remove(utf8_token.begin(), utf8_token.end(), U' '), utf8_token.end());

        for (int i = 0; i < utf8_token.length(); i++) {
          if (i == utf8_token.length() - 1) {
            std::string boundary(1, utf8_token[i]);
            byte_list.emplace_back(bbpe_encoder_.GetTokenId(boundary + "</w>"), 1);
          } else {
            byte_list.emplace_back(byte_encoder_[static_cast<unsigned char>(utf8_token[i])], 1);
          }
        }
      } else {
        for (char& cp : utf8_token) {
          byte_list.emplace_back(byte_encoder_[static_cast<unsigned char>(cp)], 1);
        }
      }

      // Perform BPE
      bbpe_encoder_.PerformBPE(byte_list);

      // Add output to result
      for (auto p : byte_list) {
        if (static_cast<int64_t>(res.size()) >= max_length) {
          break;
        }

        res.push_back(p.first);

        if (compute_offset_mapping) {
          if (clean_up_spaces) {
            offset_mapping.emplace_back(std::make_pair(offset, gsl::narrow<size_t>(offset + p.second)));
            offset += p.second;
          } else {
            offset_mapping.emplace_back(std::make_pair(offset, gsl::narrow<size_t>(offset + (size_t)p.second + space_dif)));
            offset += ((size_t)p.second + space_dif);
          }
        }
      }
    }

    if (compute_offset_mapping) {
      if (ModelName() != kModel_GPT2) {
        // Add offset mapping for EOS token
        offset_mapping.emplace_back(std::make_pair(0, 0));
      }
      // Add offset mappings for input in this instance to list of offset mappings for all inputs
      offset_map.emplace_back(offset_mapping);
    }
  }

  if (ModelName() != kModel_GPT2 && ModelName() != kModel_CodeGen) {
    // Add EOS token to result
    res.push_back(eos_token_id_);
  }

  return res;
}

TfmStatus BPETokenizer::Encode(std::string_view sv_input, std::vector<tfmTokenId_t>& ids) const {
  std::list<OffsetMappingType> offset_map;
  auto max_length = padding_length_ < 0 ? std::numeric_limits<uint32_t>::max() : padding_length_;
  std::vector<tfmTokenId_t> res = Encode(sv_input,
                                         max_length,
                                         false,
                                         offset_map);
  ids = res;
  return {};
}

TfmStatus BPETokenizer::Decode(const span<tfmTokenId_t const>& ids, std::string& text) const {
  bool f_special_last = false;
  bool f_special = false;
  auto count = static_cast<size_t>(ids.size());
  auto p_ids = ids.data();

  auto& args = decode_extra_args_;
  auto end_word_suffix = bbpe_encoder_.GetEndWordSuffix();
  for (size_t tok_idx = 0; tok_idx < count; ++tok_idx) {
    const auto token = *(p_ids + tok_idx);
    std::string decoded_token;
    f_special = all_special_ids_.count(token) ? true : false;
    if (args.skip_special_tokens_ && f_special) {
      f_special_last = f_special;
      continue;
    }

    if (added_tokens_.count(token)) {
      const std::string ws = added_tokens_.at(token);
      decoded_token = (std::string)ws;
    } else if (static_cast<size_t>(token) < arr_vocab_.size()) {
      const auto str = FromUTF8(arr_vocab_[token]);
      for (unsigned char wchr : str) {
        if (byte_decoder_.count(wchr) == 0 && wchr <= 0xFF) {
          // std::cout << "Error: cannot find the byte_decoder_ for " << (uint32_t)(unsigned char)wchr << std::endl;
          decoded_token.push_back(gsl::narrow<unsigned char>(wchr));
        } else {
          unsigned char uchr = byte_decoder_.at(wchr);
          decoded_token.push_back(uchr);
        }
      }
    } else {
      if (args.skip_special_tokens_) {
        continue;
      } else {
        decoded_token = unk_token_;
      }
    }

    // remove the end_word_suffix like </w> or </s> etc.
    if (end_word_suffix.size() > 0) {
      if (decoded_token.size() >= end_word_suffix.size() &&
          decoded_token.substr(decoded_token.size() - end_word_suffix.size()) == end_word_suffix) {
        decoded_token = decoded_token.substr(0, decoded_token.size() - end_word_suffix.size());
        decoded_token += ' ';
      }
    }

    if (args.whitespace_token_ &&
        f_special && (tok_idx > 0 && !f_special_last)) {
      text.push_back(' ');
    }

    text.append(decoded_token);

    if (args.whitespace_token_ &&
        f_special && tok_idx != count - 1) {
      text.push_back(' ');
    }

    f_special_last = f_special;
  }

  if (!text.empty() && text.back() == ' ') {
    text.pop_back();
  }

  return {};
}

}  // namespace tfm
