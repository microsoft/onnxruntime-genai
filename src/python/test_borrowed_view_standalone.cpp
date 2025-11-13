// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Standalone test for borrowed views without GTest dependency
// This allows us to run valgrind memory leak detection

#include "wrappers/oga_wrappers.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace OgaPy;

// Simple assertion with message
#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "ASSERTION FAILED: " << msg << std::endl; \
        std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Mock parent for testing
struct MockParent : OgaObject {
    MockParent(const std::vector<int32_t>& data) : data_(data) {}
    ~MockParent() override {
        std::cout << "MockParent destroyed" << std::endl;
    }
    
    const int32_t* data() const { return data_.data(); }
    size_t size() const { return data_.size(); }
    
private:
    std::vector<int32_t> data_;
};

void test_basic_construction() {
    std::cout << "\n=== Test: Basic Construction ===" << std::endl;
    
    std::vector<int32_t> test_data = {1, 2, 3, 4, 5};
    auto parent = new MockParent(test_data);
    
    {
        BorrowedArrayView<MockParent, int32_t> view(parent, parent->data(), parent->size());
        
        ASSERT(view.size() == 5, "Size should be 5");
        ASSERT(view[0] == 1, "First element should be 1");
        ASSERT(view[4] == 5, "Last element should be 5");
    }
    
    delete parent;
    std::cout << "PASSED" << std::endl;
}

void test_parent_kept_alive() {
    std::cout << "\n=== Test: Parent Kept Alive ===" << std::endl;
    
    std::vector<int32_t> test_data = {10, 20, 30};
    auto parent = new MockParent(test_data);
    
    auto view = new BorrowedArrayView<MockParent, int32_t>(parent, parent->data(), parent->size());
    
    // "Release" our handle to parent - view should keep it alive
    MockParent* parent_ptr = parent;
    parent = nullptr;
    
    ASSERT(view->size() == 3, "View should still have size 3");
    ASSERT((*view)[1] == 20, "View should still access data");
    
    std::cout << "About to delete view (should destroy parent)..." << std::endl;
    delete view;  // Should trigger parent destruction
    
    std::cout << "PASSED" << std::endl;
}

void test_multiple_views() {
    std::cout << "\n=== Test: Multiple Views ===" << std::endl;
    
    std::vector<int32_t> test_data = {100, 200, 300};
    auto parent = new MockParent(test_data);
    
    auto view1 = new BorrowedArrayView<MockParent, int32_t>(parent, parent->data(), parent->size());
    auto view2 = new BorrowedArrayView<MockParent, int32_t>(parent, parent->data(), parent->size());
    auto view3 = new BorrowedArrayView<MockParent, int32_t>(parent, parent->data(), parent->size());
    
    std::cout << "Created 3 views, deleting parent..." << std::endl;
    delete parent;
    parent = nullptr;
    
    ASSERT(view1->size() == 3, "View1 still valid");
    ASSERT(view2->size() == 3, "View2 still valid");
    ASSERT(view3->size() == 3, "View3 still valid");
    
    std::cout << "Deleting views one by one..." << std::endl;
    delete view1;
    std::cout << "  View1 deleted" << std::endl;
    
    ASSERT(view2->size() == 3, "View2 still valid after view1 deleted");
    
    delete view2;
    std::cout << "  View2 deleted" << std::endl;
    
    ASSERT(view3->size() == 3, "View3 still valid");
    
    delete view3;
    std::cout << "  View3 deleted (parent should be destroyed now)" << std::endl;
    
    std::cout << "PASSED" << std::endl;
}

void test_iterator_support() {
    std::cout << "\n=== Test: Iterator Support ===" << std::endl;
    
    std::vector<int32_t> test_data = {5, 10, 15, 20};
    auto parent = new MockParent(test_data);
    
    BorrowedArrayView<MockParent, int32_t> view(parent, parent->data(), parent->size());
    
    std::vector<int32_t> result;
    for (const auto& val : view) {
        result.push_back(val);
    }
    
    ASSERT(result.size() == test_data.size(), "Result size matches");
    for (size_t i = 0; i < result.size(); ++i) {
        ASSERT(result[i] == test_data[i], "Iterator values match");
    }
    
    delete parent;
    std::cout << "PASSED" << std::endl;
}

void test_with_real_sequences() {
    std::cout << "\n=== Test: Real OgaSequences ===" << std::endl;
    
    ::OgaSequences* raw_sequences = nullptr;
    auto result = OgaCreateSequences(&raw_sequences);
    ASSERT(result == nullptr, "Failed to create sequences");
    
    auto sequences = new OgaPy::OgaSequences(raw_sequences);
    
    std::vector<int32_t> tokens = {100, 200, 300, 400};
    result = OgaAppendTokenSequence(tokens.data(), tokens.size(), sequences->get());
    ASSERT(result == nullptr, "Failed to append tokens");
    
    // Use the GetSequenceData method
    auto view = sequences->GetSequenceData(0);
    
    ASSERT(view->size() == tokens.size(), "View size matches");
    for (size_t i = 0; i < tokens.size(); ++i) {
        ASSERT((*view)[i] == tokens[i], "Token values match");
    }
    
    std::cout << "Deleting sequences (view should keep it alive)..." << std::endl;
    delete sequences;
    
    ASSERT(view->size() == tokens.size(), "View still valid after sequences deleted");
    
    std::cout << "Deleting view (should free sequences now)..." << std::endl;
    delete view;
    
    std::cout << "PASSED" << std::endl;
}

void test_rapid_cycles() {
    std::cout << "\n=== Test: Rapid Create/Delete Cycles ===" << std::endl;
    
    std::vector<int32_t> test_data = {1, 2, 3};
    auto parent = new MockParent(test_data);
    
    for (int i = 0; i < 1000; ++i) {
        auto view = new BorrowedArrayView<MockParent, int32_t>(parent, parent->data(), parent->size());
        ASSERT(view->size() == 3, "View size correct");
        delete view;
    }
    
    delete parent;
    std::cout << "PASSED (1000 cycles)" << std::endl;
}

int main() {
    std::cout << "===================================" << std::endl;
    std::cout << "Borrowed View Memory Safety Tests" << std::endl;
    std::cout << "===================================" << std::endl;
    
    try {
        test_basic_construction();
        test_parent_kept_alive();
        test_multiple_views();
        test_iterator_support();
        test_with_real_sequences();
        test_rapid_cycles();
        
        std::cout << "\n===================================" << std::endl;
        std::cout << "ALL TESTS PASSED ✓" << std::endl;
        std::cout << "===================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
