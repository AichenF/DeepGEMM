  // Logical/Physical ElementA per Chunk
  using LogicalElemsAPerChunk = conditional_t<IsTF32, _2, _4>;
  using PhysicalElemsAPerChunk = Int<LogicalElemsAPerChunk{} / ElementASparsity{}>;

  // Metadata Bits
  using ElementEBitsPerChunk = _4;
  using ElementEBitsPerElementAMma = cute::conditional_t<IsTF32, _4, _2>;

  struct MetadataOneChunk2to4{

    CUTE_DEVICE
    void set_metadata_bits(int elt_log_idx, int elt_phy_idx) {
      auto metadata_bits = [&]() -> uint8_t {
        CUTLASS_ASSERT(elt_log_idx >= 0 && elt_log_idx < 4);
        switch (elt_log_idx) {
          case 0:
            return 0b00;
          case 1:
            return 0b01;
          case 2:
            return 0b10;
          case 3:
            return 0b11;
          default:
            CUTLASS_ASSERT(false);
            CUTE_GCC_UNREACHABLE;
            return 0b00;
        }
      };

      storage_ |= (metadata_bits() << (2 * elt_phy_idx));
    }

    CUTE_DEVICE
    ElementEChunk storage() const {
      return ElementEChunk{storage_};
    }

  private:
    uint8_t storage_ = 0b0000;
  };

// SPARSE GMMA 64x128x64 TN F32+=E4M3*E4M3
template <
  GMMA::ScaleIn  scaleA = GMMA::ScaleIn::One,
  GMMA::ScaleIn  scaleB = GMMA::ScaleIn::One,
  GMMA::SparseSel spsel = GMMA::SparseSel::Zero
>
struct GMMA_64x128x64_F32E4M3E4M3_SS_TN
{
  using DRegisters = void;
  using ARegisters = uint64_t[1];
  using ERegisters = uint32_t[1];
  using BRegisters = uint64_t[1];
  using CRegisters = float[64];

  CUTE_HOST_DEVICE static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      float         & d00, float         & d01, float         & d02, float         & d03,
      float         & d04, float         & d05, float         & d06, float         & d07,
      float         & d08, float         & d09, float         & d10, float         & d11,
      float         & d12, float         & d13, float         & d14, float         & d15,
      float         & d16, float         & d17, float         & d18, float         & d19,
      float         & d20, float         & d21, float         & d22, float         & d23,
      float         & d24, float         & d25, float         & d26, float         & d27,
      float         & d28, float         & d29, float         & d30, float         & d31,
      float         & d32, float         & d33, float         & d34, float         & d35,
      float         & d36, float         & d37, float         & d38, float         & d39,
      float         & d40, float         & d41, float         & d42, float         & d43,
      float         & d44, float         & d45, float         & d46, float         & d47,
      float         & d48, float         & d49, float         & d50, float         & d51,
      float         & d52, float         & d53, float         & d54, float         & d55,
      float         & d56, float         & d57, float         & d58, float         & d59,
      float         & d60, float         & d61, float         & d62, float         & d63,
      uint32_t const& e,
      GMMA::ScaleOut const scale_D = GMMA::ScaleOut::One)
