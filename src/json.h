// JSON Parser
//
// Just implement the JSON::Element structure per JSON element you're interested in (note it's per element,
// so a JSON tree structure requires a tree of JSON::Element objects)
// Then call JSON::Parse passing it the root JSON element object, and a string_view of the JSON data
//
// For the elements inside of an array, the names will be empty strings
// The root element also has no names.
//
namespace JSON {
struct Element {
  virtual void OnComplete(bool empty) {}  // Called when parsing for this element is finished (empty is true when it's an empty element)

  virtual void OnString(std::string_view name, std::string_view value) {}
  virtual void OnNumber(std::string_view name, double value) {}
  virtual void OnBool(std::string_view name, bool value) {}
  virtual void OnNull(std::string_view name) {}

  virtual Element& OnArray(std::string_view name);   // Default behavior ignores all elements
  virtual Element& OnObject(std::string_view name);  // Default behavior ignores all elements
};

void Parse(Element& element, std::string_view document);
}  // namespace JSON
