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
struct unknown_value_error {};  // Throw this from any Element callback to throw a std::runtime error reporting the unknown value name

struct Element {
  virtual void OnComplete(bool empty) {}  // Called when parsing for this element is finished (empty is true when it's an empty element)

  virtual void OnString(std::string_view name, std::string_view value) { throw unknown_value_error{}; }
  virtual void OnNumber(std::string_view name, double value) { throw unknown_value_error{}; }
  virtual void OnBool(std::string_view name, bool value) { throw unknown_value_error{}; }
  virtual void OnNull(std::string_view name) { throw unknown_value_error{}; }

  virtual Element& OnArray(std::string_view name);
  virtual Element& OnObject(std::string_view name);
};

void Parse(Element& element, std::string_view document);
}  // namespace JSON
