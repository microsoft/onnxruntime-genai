namespace details {
template <class P>
constexpr auto AsSpanImpl(P* p, size_t s) {
  return gsl::span<P>(p, s);
}
}  // namespace details

template <class C>
constexpr auto AsSpan(C& c) {
  return details::AsSpanImpl(c.data(), c.size());
}

template <class C>
constexpr auto AsSpan(const C& c) {
  return details::AsSpanImpl(c.data(), c.size());
}

template <class C>
constexpr auto AsSpan(C&& c) {
  return details::AsSpanImpl(c.data(), c.size());
}

template <class T>
constexpr auto AsSpan(std::initializer_list<T> c) {
  return details::AsSpanImpl(c.begin(), c.size());
}

template <class T, size_t N>
constexpr auto AsSpan(T (&arr)[N]) {
  return details::AsSpanImpl(arr, N);
}

template <class T, size_t N>
constexpr auto AsSpan(const T (&arr)[N]) {
  return details::AsSpanImpl(arr, N);
}

template <class T>
inline gsl::span<const T> EmptySpan() { return gsl::span<const T>(); }

template <class U, class T>
[[nodiscard]] inline gsl::span<U> ReinterpretAsSpan(gsl::span<T> src) {
  // adapted from gsl-lite span::as_span():
  // https://github.com/gsl-lite/gsl-lite/blob/4720a2980a30da085b4ddb4a0ea2a71af7351a48/include/gsl/gsl-lite.hpp#L4102-L4108
  Expects(src.size_bytes() % sizeof(U) == 0);
  return gsl::span<U>(reinterpret_cast<U*>(src.data()), src.size_bytes() / sizeof(U));
}

template <class T1, size_t Extent1, class T2, size_t Extent2>
[[nodiscard]] inline bool SpanEq(gsl::span<T1, Extent1> a, gsl::span<T2, Extent2> b) {
  static_assert(std::is_same_v<std::remove_const_t<T1>, std::remove_const_t<T2>>,
                "T1 and T2 should be the same type except for const qualification");
  return std::equal(a.begin(), a.end(), b.begin(), b.end());
}