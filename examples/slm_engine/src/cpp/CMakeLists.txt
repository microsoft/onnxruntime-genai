
add_library(
    slmengine
    slm_engine.cpp
    input_decoder.cpp
)

target_link_libraries(slmengine ort-genai ort)

add_executable(unit-test input_decoder.cpp input_decoder_test.cpp)
target_link_libraries(unit-test GTest::gtest)

add_executable(googletest gtest_main.cpp)
target_link_libraries(googletest GTest::gtest)

add_executable(slm-runner slm_runner.cpp)
target_link_libraries(slm-runner slmengine ort-genai ort)

add_executable(slm-server slm_server.cpp)
target_link_libraries(slm-server slmengine ort-genai ort)
    
install(TARGETS slm-runner DESTINATION bin)
install(TARGETS slm-server DESTINATION bin)
