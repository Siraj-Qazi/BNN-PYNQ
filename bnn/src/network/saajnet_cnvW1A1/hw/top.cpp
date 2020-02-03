
#include "config.h"
#include "bnn-library.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM>  weights1;
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM>  weights2;
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM>  weights3;
static BinaryWeights<L4_SIMD, L4_PE, L4_WMEM>  weights4;
static BinaryWeights<L5_SIMD, L5_PE, L5_WMEM>  weights5;
static BinaryWeights<L6_SIMD, L6_PE, L6_WMEM>  weights6;
static BinaryWeights<L7_SIMD, L7_PE, L7_WMEM>  weights7;
static BinaryWeights<L8_SIMD, L8_PE, L8_WMEM>  weights8;
static BinaryWeights<L9_SIMD, L9_PE, L9_WMEM>  weights9;
static BinaryWeights<L10_SIMD, L10_PE, L10_WMEM>  weights10;
static BinaryWeights<L11_SIMD, L11_PE, L11_WMEM>  weights11;

static ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs2;
static ThresholdsActivation<L3_TMEM, L3_PE, L3_API, ap_int<16>, ap_uint<L3_API>>  		threshs3;
static ThresholdsActivation<L4_TMEM, L4_PE, L4_API, ap_int<16>, ap_uint<L4_API>>  		threshs4;
static ThresholdsActivation<L5_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs5;
static ThresholdsActivation<L6_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs6;
static ThresholdsActivation<L7_TMEM, L7_PE, L7_API, ap_int<16>, ap_uint<L7_API>>  		threshs7;
static ThresholdsActivation<L8_TMEM, L8_PE, L8_API, ap_int<16>, ap_uint<L8_API>>  		threshs8;
static ThresholdsActivation<L9_TMEM, L9_PE, L9_API, ap_int<16>, ap_uint<L9_API>>  		threshs9;
static ThresholdsActivation<L10_TMEM, L10_PE, L10_API, ap_int<16>, ap_uint<L10_API>>  		threshs10;

unsigned int paddedSizeHW(unsigned int in, unsigned int padTo) 
{
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) 
{
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 1:
      threshs0.m_thresholds[targetMem][targetInd][targetThresh] = *reinterpret_cast<ap_fixed<64, 56> *>(&val);
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 3:
      threshs1.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 5:
      threshs2.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 7:
      threshs3.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 8:
	  weights4.m_weights[targetMem][targetInd] = val;
	  break;
	case 9:
	  threshs4.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 10:
	  weights5.m_weights[targetMem][targetInd] = val;
	  break;
	case 11:
	  threshs5.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 12:
	  weights6.m_weights[targetMem][targetInd] = val;
	  break;
	case 13:
	  threshs6.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 14:
	  weights7.m_weights[targetMem][targetInd] = val;
	  break;
	case 15:
      threshs7.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 16:
	  weights8.m_weights[targetMem][targetInd] = val;
	  break;
	case 17:
	  threshs8.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 18:
	  weights9.m_weights[targetMem][targetInd] = val;
	  break;
	case 19:
      threshs9.m_thresholds[targetMem][targetInd][targetThresh] = val;
      break;
    case 20:
	  weights10.m_weights[targetMem][targetInd] = val;
	  break;
	case 21:
	  threshs10.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 22:
	  weights11.m_weights[targetMem][targetInd] = val;
	  break;
	case 23:
	  //do nothing
	  break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {
#pragma HLS DATAFLOW

	stream<ap_uint<64> > instream("DoCompute.instream");
	stream<ap_uint<8*IMG_CH> > instream_bitw("DoCompute.instream_bitw");
#pragma HLS STREAM variable=instream_bitw depth=128

	// first conv 3x3
	stream<ap_uint<L0_OFM_CH> > convstream0("DoCompute.convstream0");
	stream<ap_uint<L0_OFM_CH> > poolstream0("DoCompute.poolstream0");
#pragma HLS STREAM variable=poolstream0 depth=128

	// cloning streams
	stream<ap_uint<L0_OFM_CH>> poolstream0copy1("DoCompute.poolstream0copy1");
#pragma HLS STREAM variable=poolstream0copy1 depth=128
	stream<ap_uint<L0_OFM_CH>> poolstream0copy2("DoCompute.poolstream0copy2");
#pragma HLS STREAM variable=poolstream0copy2 depth=128
	stream<ap_uint<L0_OFM_CH>> poolstream0copy3("DoCompute.poolstream0copy3");
#pragma HLS STREAM variable=poolstream0copy3 depth=128
	
	// incep 1x1
	stream<ap_uint<L1_OFM_CH> > convstream1a("DoCompute.convstream1a");
#pragma HLS STREAM variable=convstream1a depth=128

	// incep 3x3
	stream<ap_uint<L2_OFM_CH> > convstream1b("DoCompute.convstream1b");
#pragma HLS STREAM variable=convstream1b depth=128

	// incep 5x5
	stream<ap_uint<L3_OFM_CH> > convstream1c("DoCompute.convstream1c");
#pragma HLS STREAM variable=convstream1c depth=128

	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>> concatstream1("DoCompute.concatstream1");
#pragma HLS STREAM variable=concatstream1 depth=128

	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH> > poolstream1("DoCompute.poolstream1");
#pragma HLS STREAM variable=poolstream1 depth=128

	// cloning streams
	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>> poolstream1copy1("DoCompute.poolstream1copy1");
#pragma HLS STREAM variable=poolstream1copy1 depth=128
	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>> poolstream1copy2("DoCompute.poolstream1copy2");
#pragma HLS STREAM variable=poolstream1copy2 depth=128
	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>> poolstream1copy3("DoCompute.poolstream1copy3");
#pragma HLS STREAM variable=poolstream1copy3 depth=128

	// incep 1x1
	stream<ap_uint<L4_OFM_CH> > convstream2a("DoCompute.convstream2a");
#pragma HLS STREAM variable=convstream2a depth=128

	// incep 3x3
	stream<ap_uint<L5_OFM_CH> > convstream2b("DoCompute.convstream2b");
#pragma HLS STREAM variable=convstream2b depth=128

	// incep 5x5
	stream<ap_uint<L6_OFM_CH> > convstream2c("DoCompute.convstream2c");
#pragma HLS STREAM variable=convstream2c depth=128

	stream<ap_uint<L4_OFM_CH+L5_OFM_CH+L6_OFM_CH>> concatstream2("DoCompute.concatstream2");
#pragma HLS STREAM variable=concatstream2 depth=128

	stream<ap_uint<L4_OFM_CH+L5_OFM_CH+L6_OFM_CH> > poolstream2("DoCompute.poolstream2");
#pragma HLS STREAM variable=poolstream2 depth=128


	// cloning streams
	stream<ap_uint<L4_OFM_CH+L5_OFM_CH+L6_OFM_CH>> poolstream2copy1("DoCompute.poolstream2copy1");
#pragma HLS STREAM variable=poolstream2copy1 depth=128
	stream<ap_uint<L4_OFM_CH+L5_OFM_CH+L6_OFM_CH>> poolstream2copy2("DoCompute.poolstream2copy2");
#pragma HLS STREAM variable=poolstream2copy2 depth=128
	stream<ap_uint<L4_OFM_CH+L5_OFM_CH+L6_OFM_CH>> poolstream2copy3("DoCompute.poolstream2copy3");
#pragma HLS STREAM variable=poolstream2copy3 depth=128

	// incep 1x1
	stream<ap_uint<L7_OFM_CH> > convstream3a("DoCompute.convstream3a");
#pragma HLS STREAM variable=convstream3a depth=128

	// incep 3x3
	stream<ap_uint<L8_OFM_CH> > convstream3b("DoCompute.convstream3b");
#pragma HLS STREAM variable=convstream3b depth=128

	// incep 5x5
	stream<ap_uint<L9_OFM_CH> > convstream3c("DoCompute.convstream3c");
#pragma HLS STREAM variable=convstream3c depth=128

    // concat L7 + L8 + L9
	stream<ap_uint<L7_OFM_CH+L8_OFM_CH+L9_OFM_CH>> concatstream3("DoCompute.concatstream3");
#pragma HLS STREAM variable=concatstream3 depth=128

   // 3rd max pooling  
	stream<ap_uint<L7_OFM_CH+L8_OFM_CH+L9_OFM_CH> > poolstream3("DoCompute.poolstream3");
#pragma HLS STREAM variable=poolstream3 depth=128


	stream<ap_uint<64> > fcstream1("DoCompute.fcstream1");
#pragma HLS STREAM variable=fcstream1 depth=128

	stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");

	const unsigned int inBits = IMG_DIM*IMG_DIM*IMG_CH*8;
	const unsigned int outBits = L11_MH*16;

	Mem2Stream_Batch<64, inBits/8> (in, instream, numReps);
	StreamingDataWidthConverter_Batch<64, 8, (inBits) / 64> (instream, instream_bitw, numReps);

	// convolutional layers
	ConvLayerValid_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE, Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Identity, Recast<Binary>>(instream_bitw, convstream0, weights0, threshs0, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L0_OFM_DIM, 2, L0_OFM_CH>(convstream0, poolstream0, numReps);

	// cloning one input stream into 3
	CloneStream_Batch<L0_OFM_CH, L1_IFM_DIM>(poolstream0, poolstream0copy1, poolstream0copy2, poolstream0copy3, numReps);
	
	// 1x1 conv block
	ConvLayerValid_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, Recast<XnorMul>>(poolstream0copy1, convstream1a, weights1, threshs1, numReps, ap_resource_lut());
	// 3x3 conv block
	ConvLayerSame_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, Recast<XnorMul>>(poolstream0copy2, convstream1b, weights2, threshs2, numReps, ap_resource_lut());
	// 5x5 conv block
	ConvLayerSame_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, Recast<XnorMul>>(poolstream0copy3, convstream1c, weights3, threshs3, numReps, ap_resource_lut());
	//concatenating  convstream1, convstream2 and convstream3
	ConcatStream_Batch<L3_OFM_CH, L3_OFM_DIM>(convstream1a, convstream1b, convstream1c, concatstream1, numReps);

	StreamingMaxPoolEven_Batch<L3_OFM_DIM, 2, L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>(concatstream1, poolstream1, numReps);


	// cloning one input stream into 3
	CloneStream_Batch<L3_OFM_CH * 3, L4_IFM_DIM>(poolstream1, poolstream1copy1, poolstream1copy2, poolstream1copy3, numReps);
	// 1x1 conv block
	ConvLayerValid_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, L4_SIMD, L4_PE, Recast<XnorMul>>(poolstream1copy1, convstream2a, weights4, threshs4, numReps, ap_resource_lut());
	// 3x3 conv block
	ConvLayerSame_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM, L5_SIMD, L5_PE, Recast<XnorMul>>(poolstream1copy2, convstream2b, weights5, threshs5, numReps, ap_resource_lut());
	// 5x5 conv block
	ConvLayerSame_Batch<L6_K, L6_IFM_CH, L6_IFM_DIM, L6_OFM_CH, L6_OFM_DIM, L6_SIMD, L6_PE, Recast<XnorMul>>(poolstream1copy3, convstream2c, weights6, threshs6, numReps, ap_resource_lut());
	//concatenating  convstream1, convstream2 and convstream3
	ConcatStream_Batch<L6_OFM_CH, L6_OFM_DIM>(convstream2a, convstream2b, convstream2c, concatstream2, numReps);

	StreamingMaxPoolEven_Batch<L6_OFM_DIM, 2, L4_OFM_CH+L5_OFM_CH+L6_OFM_CH>(concatstream2, poolstream2, numReps);

	// cloning one input stream into 3
	CloneStream_Batch<L6_OFM_CH*3, L7_IFM_DIM>(poolstream2, poolstream2copy1, poolstream2copy2, poolstream2copy3, numReps);
	// 1x1 conv block
	ConvLayerValid_Batch<L7_K, L7_IFM_CH, L7_IFM_DIM, L7_OFM_CH, L7_OFM_DIM, L7_SIMD, L7_PE, Recast<XnorMul>>(poolstream2copy1, convstream3a, weights7, threshs7, numReps, ap_resource_lut());
	// 3x3 conv block
	ConvLayerSame_Batch<L8_K, L8_IFM_CH, L8_IFM_DIM, L8_OFM_CH, L8_OFM_DIM, L8_SIMD, L8_PE, Recast<XnorMul>>(poolstream2copy2, convstream3b, weights8, threshs8, numReps, ap_resource_lut());
	// 5x5 conv block
	ConvLayerSame_Batch<L9_K, L9_IFM_CH, L9_IFM_DIM, L9_OFM_CH, L9_OFM_DIM, L9_SIMD, L9_PE, Recast<XnorMul>>(poolstream2copy3, convstream3c, weights9, threshs9, numReps, ap_resource_lut());
	//concatenating  convstream1, convstream2 and convstream3
	ConcatStream_Batch<L9_OFM_CH, L9_OFM_DIM>(convstream3a, convstream3b, convstream3c, concatstream3, numReps);

	StreamingMaxPoolEven_Batch<L9_OFM_DIM, 2, L7_OFM_CH+L8_OFM_CH+L9_OFM_CH>(concatstream3, poolstream3, numReps);


	// fully connected layers
	WidthAdjustedOutputStream<L11_PE, 64, L11_MH / L11_PE>  wa_out(memOutStrm, numReps);
	StreamingFCLayer_Batch<L10_MW, L10_MH, L10_SIMD, L10_PE, Recast<XnorMul>>
	(poolstream3, fcstream1,  weights10, threshs10, numReps, ap_resource_lut());
	StreamingFCLayer_Batch<L11_MW, L11_MH, L11_SIMD, L11_PE, Recast<XnorMul>, Slice<ap_uint<16> >>
	(fcstream1, static_cast<hls::stream<ap_uint<L11_PE>>&>(wa_out), weights11, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_lut());

  Stream2Mem_Batch<64, outBits/8>(memOutStrm, out, numReps);

}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=512
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs1.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs2.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs3.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights4.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs4.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs4.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights5.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs5.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs5.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights6.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs6.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs6.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights7.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs7.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs7.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights8.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs8.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs8.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights9.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs9.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs9.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights10.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs10.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs10.m_thresholds complete dim=3
#pragma HLS ARRAY_PARTITION variable=weights11.m_weights complete dim=1

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
