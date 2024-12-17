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

struct Value : private std::variant<std::string_view, double, bool, std::nullptr_t> {
  using std::variant<std::string_view, double, bool, std::nullptr_t>::variant;
  static constexpr size_t type_count_v = std::variant_size_v<variant>;

  // This will generate a descriptive error when the types don't match
  template <typename T>
  T Get() const {
    try {
      return std::get<T>(*this);
    } catch (const std::bad_variant_access&) {
      throw type_mismatch{index(), Value{T{}}.index()};
    }
  }

  operator std::string() const { return std::string{Get<std::string_view>()}; }
  operator double() const { return Get<double>(); }
  operator float() const { return static_cast<float>(Get<double>()); }
  operator int() const { return static_cast<int>(Get<double>()); }
  operator bool() const { return Get<bool>(); }
  explicit operator char() const = delete;  // To avoid ambiguity when converting to std::string
};

struct Element {
  virtual void OnComplete(bool empty) {}  // Called when parsing for this element is finished (empty is true when it's an empty element)

  virtual void OnValue(std::string_view name, Value value) { throw unknown_value_error{}; }

  virtual Element& OnArray(std::string_view name) { throw unknown_value_error{}; }
  virtual Element& OnObject(std::string_view name) { throw unknown_value_error{}; }
};

void Parse(Element& element, std::string_view document);
}  // namespace JSON
