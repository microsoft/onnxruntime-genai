// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

namespace Generators {

// Sizing and addressing for the ring of blocks that serves a sliding-window paged layer.
//
// Such a layer never reads further back than `window_size` positions, so it does not need one
// block per position. It is given a short ring instead, and the block table simply repeats that
// ring across its columns. The operator is unchanged: it indexes the table with the token's true
// position, so position p lands in ring slot p % ring_slots and positions that have fallen out of
// the window are overwritten in place.

// Positions the ring must hold at once. A step covers [past, past + scheduled) and every one of
// those queries reaches back `window_size` positions, so the oldest position still being read is
// `past - (window_size - 1)`.
constexpr size_t WindowLiveSpan(size_t chunk_size, size_t window_size) {
  return chunk_size + window_size - 1;
}

// Blocks needed to cover that live span. `chunk_size` bounds `scheduled`, so a prefill has to be
// chunked for this to be smaller than the context length.
constexpr size_t WindowRingBlocks(size_t chunk_size, size_t window_size, size_t block_size) {
  return (WindowLiveSpan(chunk_size, window_size) + block_size - 1) / block_size;
}

// Block table column j names ring block j % ring_blocks.
constexpr size_t WindowRingColumn(size_t column, size_t ring_blocks) {
  return column % ring_blocks;
}

// Ring slot the operator ends up reading for a position, following the table it was given:
// column p / block_size selects the block, p % block_size the offset inside it.
constexpr size_t WindowRingSlot(size_t position, size_t ring_blocks, size_t block_size) {
  return WindowRingColumn(position / block_size, ring_blocks) * block_size + position % block_size;
}

}  // namespace Generators
