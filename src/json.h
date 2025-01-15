// JSON Parser
//
// Just implement the JSON::Element structure per JSON element you're interested in (note it's per element,
// so a JSON tree structure requires a tree of JSON::Element objects)
// Then call JSON::Parse with the root JSON element object, and a string_view of the JSON data
//
// For the elements inside of an array, the names will be empty strings
// The root element also has no names.
//
namespace JSON {
struct unknown_value_error : std::exception {};  // Throw this from any Element callback to throw a std::runtime error reporting the unknown value name
struct type_mismatch {                           // When a file has one type, but we're expecting another type. "seen" & "expected" are indices into the Value std::variant below
  size_t seen, expected;
};

using Value = std::variant<std::string_view, double, bool, std::nullptr_t>;

// To see descriptive errors when types don't match, use this instead of std::get
template <typename T>
T Get(Value& var) {
  try {
    return std::get<T>(var);
  } catch (const std::bad_variant_access&) {
    throw type_mismatch{var.index(), Value{T{}}.index()};
  }
}

struct Element {
  virtual void OnComplete(bool empty) {}  // Called when parsing for this element is finished (empty is true when it's an empty element)

  virtual void OnValue(std::string_view name, Value value) { throw unknown_value_error{}; }

  virtual Element& OnArray(std::string_view name) { throw unknown_value_error{}; }
  virtual Element& OnObject(std::string_view name) { throw unknown_value_error{}; }
};

void Parse(Element& element, std::string_view document);
}  // namespace JSON
