// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include "../wrappers/oga_wrappers.h"
#include <vector>
#include <memory>

using namespace OgaPy;

// Mock parent class for testing
struct MockParent : OgaObject {
  MockParent(const std::vector<int32_t>& data) : data_(data) {}
  ~MockParent() override = default;
  
  const int32_t* data() const { return data_.data(); }
  size_t size() const { return data_.size(); }
  
private:
  std::vector<int32_t> data_;
};

class BorrowedArrayViewTest : public ::testing::Test {
protected:
  void SetUp() override {
    test_data_ = {1, 2, 3, 4, 5};
    parent_ = new MockParent(test_data_);
  }

  void TearDown() override {
    // Parent should be kept alive by views, or cleaned up if no views exist
    if (parent_ref_count_ == 0) {
      delete parent_;
    }
  }

  std::vector<int32_t> test_data_;
  MockParent* parent_ = nullptr;
  int parent_ref_count_ = 0;
};

// Test basic construction and data access
TEST_F(BorrowedArrayViewTest, BasicConstruction) {
  BorrowedArrayView<MockParent, int32_t> view(parent_, parent_->data(), parent_->size());
  
  EXPECT_EQ(view.size(), 5);
  EXPECT_EQ(view.data(), parent_->data());
  EXPECT_EQ(view[0], 1);
  EXPECT_EQ(view[4], 5);
}

// Test that parent is kept alive
TEST_F(BorrowedArrayViewTest, ParentKeptAlive) {
  auto view = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  
  // Manually track that parent has a reference
  parent_ref_count_ = 1;
  
  // View should have incremented parent's ref count
  // Even if we "release" our handle to parent, view keeps it alive
  MockParent* parent_ptr = parent_;
  const int32_t* data_ptr = view->data();
  
  // Data should still be accessible through view
  EXPECT_EQ(view->size(), 5);
  EXPECT_EQ(view->data(), data_ptr);
  
  delete view;
  parent_ref_count_ = 0;
}

// Test element access operator
TEST_F(BorrowedArrayViewTest, ElementAccess) {
  BorrowedArrayView<MockParent, int32_t> view(parent_, parent_->data(), parent_->size());
  
  for (size_t i = 0; i < test_data_.size(); ++i) {
    EXPECT_EQ(view[i], test_data_[i]);
  }
}

// Test out of bounds access
TEST_F(BorrowedArrayViewTest, OutOfBoundsAccess) {
  BorrowedArrayView<MockParent, int32_t> view(parent_, parent_->data(), parent_->size());
  
  EXPECT_THROW(view[5], std::out_of_range);
  EXPECT_THROW(view[100], std::out_of_range);
}

// Test iterator support
TEST_F(BorrowedArrayViewTest, IteratorSupport) {
  BorrowedArrayView<MockParent, int32_t> view(parent_, parent_->data(), parent_->size());
  
  std::vector<int32_t> result;
  for (auto it = view.begin(); it != view.end(); ++it) {
    result.push_back(*it);
  }
  
  EXPECT_EQ(result, test_data_);
}

// Test range-based for loop
TEST_F(BorrowedArrayViewTest, RangeBasedForLoop) {
  BorrowedArrayView<MockParent, int32_t> view(parent_, parent_->data(), parent_->size());
  
  std::vector<int32_t> result;
  for (const auto& value : view) {
    result.push_back(value);
  }
  
  EXPECT_EQ(result, test_data_);
}

// Test null parent throws
TEST_F(BorrowedArrayViewTest, NullParentThrows) {
  EXPECT_THROW(
    BorrowedArrayView<MockParent, int32_t>(nullptr, parent_->data(), parent_->size()),
    std::invalid_argument
  );
}

// Test null data with non-zero size throws
TEST_F(BorrowedArrayViewTest, NullDataNonZeroSizeThrows) {
  EXPECT_THROW(
    BorrowedArrayView<MockParent, int32_t>(parent_, nullptr, 5),
    std::invalid_argument
  );
}

// Test null data with zero size is allowed
TEST_F(BorrowedArrayViewTest, NullDataZeroSizeAllowed) {
  EXPECT_NO_THROW(
    BorrowedArrayView<MockParent, int32_t>(parent_, nullptr, 0)
  );
  
  BorrowedArrayView<MockParent, int32_t> view(parent_, nullptr, 0);
  EXPECT_EQ(view.size(), 0);
  EXPECT_EQ(view.data(), nullptr);
}

// Test move construction
TEST_F(BorrowedArrayViewTest, MoveConstruction) {
  auto view1 = BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  auto view2 = std::move(view1);
  
  EXPECT_EQ(view2.size(), 5);
  EXPECT_EQ(view2[0], 1);
  
  // view1 should be in moved-from state
  EXPECT_EQ(view1.size(), 0);
  EXPECT_EQ(view1.data(), nullptr);
}

// Test move assignment
TEST_F(BorrowedArrayViewTest, MoveAssignment) {
  std::vector<int32_t> data2 = {10, 20, 30};
  auto parent2 = new MockParent(data2);
  
  auto view1 = BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  auto view2 = BorrowedArrayView<MockParent, int32_t>(parent2, parent2->data(), parent2->size());
  
  view1 = std::move(view2);
  
  EXPECT_EQ(view1.size(), 3);
  EXPECT_EQ(view1[0], 10);
  
  // view2 should be in moved-from state
  EXPECT_EQ(view2.size(), 0);
  EXPECT_EQ(view2.data(), nullptr);
  
  delete parent2;
}

// Test that copy constructor is deleted
TEST_F(BorrowedArrayViewTest, CopyConstructorDeleted) {
  // This should not compile
  // BorrowedArrayView<MockParent, int32_t> view1(parent_, parent_->data(), parent_->size());
  // BorrowedArrayView<MockParent, int32_t> view2(view1);  // Should not compile
  
  EXPECT_FALSE(std::is_copy_constructible<BorrowedArrayView<MockParent, int32_t>>::value);
}

// Test that copy assignment is deleted
TEST_F(BorrowedArrayViewTest, CopyAssignmentDeleted) {
  EXPECT_FALSE(std::is_copy_assignable<BorrowedArrayView<MockParent, int32_t>>::value);
}

// Integration test with actual C API types using OgaSequences
class SequenceDataViewIntegrationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create a sequences object
    ::OgaSequences* raw_sequences = nullptr;
    auto result = OgaCreateSequences(&raw_sequences);
    ASSERT_EQ(result, nullptr);
    
    sequences_ = new OgaSequences(raw_sequences);
    
    // Add test data
    std::vector<int32_t> tokens = {100, 200, 300, 400};
    result = OgaAppendTokenSequence(tokens.data(), tokens.size(), sequences_->get());
    ASSERT_EQ(result, nullptr);
    
    test_tokens_ = tokens;
  }

  void TearDown() override {
    delete sequences_;
  }

  OgaSequences* sequences_ = nullptr;
  std::vector<int32_t> test_tokens_;
};

TEST_F(SequenceDataViewIntegrationTest, CreateViewFromSequences) {
  // Use the GetSequenceData method (handles ref counting automatically)
  auto view = sequences_->GetSequenceData(0);
  
  EXPECT_EQ(view->size(), test_tokens_.size());
  for (size_t i = 0; i < test_tokens_.size(); ++i) {
    EXPECT_EQ((*view)[i], test_tokens_[i]);
  }
  
  delete view;
}

TEST_F(SequenceDataViewIntegrationTest, ViewKeepsSequencesAlive) {
  // Use the GetSequenceData method
  auto view = sequences_->GetSequenceData(0);
  
  // Delete our handle to sequences (view should keep it alive)
  OgaSequences* seq_ptr = sequences_;
  sequences_ = nullptr;  // We no longer own it, view does
  
  // Data should still be accessible
  EXPECT_EQ(view->size(), test_tokens_.size());
  
  // Clean up
  delete view;
  
  // Now sequences can be deleted (view released its ref)
  // Since sequences_ is nullptr, TearDown won't try to delete it
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Additional comprehensive tests for memory leak detection

// Test: Multiple views on same parent
TEST_F(BorrowedArrayViewTest, MultipleViewsOnSameParent) {
  auto view1 = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  auto view2 = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  auto view3 = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  
  // All views should have access to same data
  EXPECT_EQ(view1->size(), test_data_.size());
  EXPECT_EQ(view2->size(), test_data_.size());
  EXPECT_EQ(view3->size(), test_data_.size());
  
  // Delete parent - all views should keep it alive
  parent_ref_count_ = 3;
  MockParent* parent_ptr = parent_;
  parent_ = nullptr;
  
  // All views still valid
  EXPECT_EQ(view1->data(), parent_ptr->data());
  EXPECT_EQ(view2->data(), parent_ptr->data());
  EXPECT_EQ(view3->data(), parent_ptr->data());
  
  // Delete views one by one
  delete view1;
  parent_ref_count_ = 2;
  
  delete view2;
  parent_ref_count_ = 1;
  
  // Last view should still work
  EXPECT_EQ(view3->size(), test_data_.size());
  
  delete view3;
  parent_ref_count_ = 0;
  
  // Now parent should be freed
  delete parent_ptr;
}

// Test: Rapid create/delete cycles
TEST_F(BorrowedArrayViewTest, RapidCreateDeleteCycles) {
  for (int i = 0; i < 100; ++i) {
    auto view = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
    EXPECT_EQ(view->size(), test_data_.size());
    delete view;
  }
}

// Test: Nested view lifetimes
TEST_F(BorrowedArrayViewTest, NestedViewLifetimes) {
  auto outer_view = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  parent_ref_count_ = 1;
  
  {
    auto inner_view = new BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
    parent_ref_count_ = 2;
    
    EXPECT_EQ(inner_view->size(), test_data_.size());
    delete inner_view;
    parent_ref_count_ = 1;
  }
  
  // Outer view still valid
  EXPECT_EQ(outer_view->size(), test_data_.size());
  delete outer_view;
  parent_ref_count_ = 0;
}

// Test: Self-assignment in move
TEST_F(BorrowedArrayViewTest, MoveAssignmentSelfAssignment) {
  auto view = BorrowedArrayView<MockParent, int32_t>(parent_, parent_->data(), parent_->size());
  
  // Self-assignment should be safe
  view = std::move(view);
  
  EXPECT_EQ(view.size(), test_data_.size());
}

// Integration test: Multiple sequences
class MultipleSequencesTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create multiple sequences
    for (int i = 0; i < 3; ++i) {
      ::OgaSequences* raw_seq = nullptr;
      ASSERT_EQ(OgaCreateSequences(&raw_seq), nullptr);
      
      auto seq = new OgaSequences(raw_seq);
      
      std::vector<int32_t> tokens;
      for (int j = 0; j < 5; ++j) {
        tokens.push_back((i + 1) * 100 + j);
      }
      
      ASSERT_EQ(OgaAppendTokenSequence(tokens.data(), tokens.size(), seq->get()), nullptr);
      
      sequences_.push_back(seq);
      test_data_.push_back(tokens);
    }
  }

  void TearDown() override {
    for (auto seq : sequences_) {
      delete seq;
    }
  }

  std::vector<OgaSequences*> sequences_;
  std::vector<std::vector<int32_t>> test_data_;
};

TEST_F(MultipleSequencesTest, ViewsFromMultipleSequences) {
  std::vector<SequenceDataView*> views;
  
  // Create views from all sequences
  for (size_t i = 0; i < sequences_.size(); ++i) {
    views.push_back(sequences_[i]->GetSequenceData(0));
  }
  
  // Verify all views
  for (size_t i = 0; i < views.size(); ++i) {
    EXPECT_EQ(views[i]->size(), test_data_[i].size());
    for (size_t j = 0; j < test_data_[i].size(); ++j) {
      EXPECT_EQ((*views[i])[j], test_data_[i][j]);
    }
  }
  
  // Delete all sequences
  for (auto seq : sequences_) {
    delete seq;
  }
  sequences_.clear();
  
  // Views should still be valid
  for (size_t i = 0; i < views.size(); ++i) {
    EXPECT_EQ(views[i]->size(), test_data_[i].size());
  }
  
  // Cleanup views
  for (auto view : views) {
    delete view;
  }
}

TEST_F(MultipleSequencesTest, InterleavedViewCreationAndDeletion) {
  std::vector<SequenceDataView*> views;
  
  // Create views in interleaved manner
  views.push_back(sequences_[0]->GetSequenceData(0));
  views.push_back(sequences_[1]->GetSequenceData(0));
  
  delete views[0];
  views.erase(views.begin());
  
  views.push_back(sequences_[2]->GetSequenceData(0));
  
  delete sequences_[1];
  sequences_[1] = nullptr;
  
  // Remaining views should still be valid
  EXPECT_EQ(views[0]->size(), test_data_[1].size());
  EXPECT_EQ(views[1]->size(), test_data_[2].size());
  
  // Cleanup
  for (auto view : views) {
    delete view;
  }
}
