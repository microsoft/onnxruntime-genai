
add_library(
    slmengine
    SHARED
    slm_engine.cpp
    input_decoder.cpp
)

target_link_libraries(slmengine ort-genai ort)

add_executable(input_decoder-test input_decoder.cpp input_decoder_test.cpp)
target_link_libraries(input_decoder-test GTest::gtest)

add_executable(
    unit-test 
    slm_engine_test.cpp
    gtest_main.cpp

)
target_link_libraries(
    unit-test  
    slmengine 
    GTest::gtest
)

add_executable(slm-runner slm_runner.cpp)
target_link_libraries(slm-runner slmengine ort-genai ort)

add_executable(slm-server slm_server.cpp)
target_link_libraries(slm-server slmengine ort-genai ort)
    
install(DIRECTORY ${ARTIFACTS}/${SYS_TARGET_ID}/include
    DESTINATION .
    FILES_MATCHING PATTERN "*.h"
)

install(FILES slm_engine.h DESTINATION include)
install(FILES input_decoder.h DESTINATION include)

install(TARGETS slmengine DESTINATION bin)
install(TARGETS slm-runner DESTINATION bin)
install(TARGETS slm-server DESTINATION bin)
install(TARGETS unit-test DESTINATION bin)
