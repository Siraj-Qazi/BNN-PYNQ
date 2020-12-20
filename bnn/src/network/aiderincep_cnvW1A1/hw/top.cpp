
#include "config.h"
#include "bnn-library.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"

static BinaryWeights<L0_SIMD, L0_PE, L0_WMEM>  weights0;     // 0th conv
static BinaryWeights<L1_SIMD, L1_PE, L1_WMEM>  weights1;     // 0th inception: 3x3 conv
static BinaryWeights<L2_SIMD, L2_PE, L2_WMEM>  weights2;	 // 0th inception: 5x5 conv
static BinaryWeights<L3_SIMD, L3_PE, L3_WMEM>  weights3;     // 0th inception: 7x7 conv
static BinaryWeights<L4_SIMD, L4_PE, L4_WMEM>  weights4;     // 1st conv
static BinaryWeights<L5_SIMD, L5_PE, L5_WMEM>  weights5;     // 2nd conv 
static BinaryWeights<L6_SIMD, L6_PE, L6_WMEM>  weights6;     // 1st inception: 3x3 conv
static BinaryWeights<L7_SIMD, L7_PE, L7_WMEM>  weights7; 	 // 1st inception: 5x5 conv
static BinaryWeights<L8_SIMD, L8_PE, L8_WMEM>  weights8;     // 1st inception: 7x7 conv
static BinaryWeights<L9_SIMD, L9_PE, L9_WMEM>  weights9;     // 3rd conv
static BinaryWeights<L10_SIMD, L10_PE, L10_WMEM>  weights10; // 4rth conv
static BinaryWeights<L11_SIMD, L11_PE, L11_WMEM>  weights11; // 2nd inception: 3x3 conv
static BinaryWeights<L12_SIMD, L12_PE, L12_WMEM>  weights12; // 2nd inception: 5x5 conv
static BinaryWeights<L13_SIMD, L13_PE, L13_WMEM>  weights13; // 2nd inception: 7x7 conv
static BinaryWeights<L14_SIMD, L14_PE, L14_WMEM>  weights14; // 5th conv
static BinaryWeights<L15_SIMD, L15_PE, L15_WMEM>  weights15; // 6th conv
static BinaryWeights<L16_SIMD, L16_PE, L16_WMEM>  weights16; // 0th fc
static BinaryWeights<L17_SIMD, L17_PE, L17_WMEM>  weights17; // 1st fc


static ThresholdsActivation<L0_TMEM, L0_PE, L0_API, ap_fixed<24, 16>, ap_uint<L0_API> > threshs0;
static ThresholdsActivation<L1_TMEM, L1_PE, L1_API, ap_int<16>, ap_uint<L1_API>>  		threshs1;
static ThresholdsActivation<L2_TMEM, L2_PE, L2_API, ap_int<16>, ap_uint<L2_API>>  		threshs2;
static ThresholdsActivation<L3_TMEM, L3_PE, L3_API, ap_int<16>, ap_uint<L3_API>>  		threshs3;
static ThresholdsActivation<L4_TMEM, L4_PE, L4_API, ap_int<16>, ap_uint<L4_API>>  		threshs4;
static ThresholdsActivation<L5_TMEM, L5_PE, L5_API, ap_int<16>, ap_uint<L5_API>>  		threshs5;
static ThresholdsActivation<L6_TMEM, L6_PE, L6_API, ap_int<16>, ap_uint<L6_API>>  		threshs6;
static ThresholdsActivation<L7_TMEM, L7_PE, L7_API, ap_int<16>, ap_uint<L7_API>>  		threshs7;
static ThresholdsActivation<L8_TMEM, L8_PE, L8_API, ap_int<16>, ap_uint<L8_API>>  		threshs8;
static ThresholdsActivation<L9_TMEM, L9_PE, L9_API, ap_int<16>, ap_uint<L9_API>>  		threshs9;
static ThresholdsActivation<L10_TMEM, L10_PE, L10_API, ap_int<16>, ap_uint<L10_API>>  		threshs10;
static ThresholdsActivation<L11_TMEM, L11_PE, L11_API, ap_int<16>, ap_uint<L11_API>>  		threshs11;
static ThresholdsActivation<L12_TMEM, L12_PE, L12_API, ap_int<16>, ap_uint<L12_API>>  		threshs12;
static ThresholdsActivation<L13_TMEM, L13_PE, L13_API, ap_int<16>, ap_uint<L13_API>>  		threshs13;
static ThresholdsActivation<L14_TMEM, L14_PE, L14_API, ap_int<16>, ap_uint<L14_API>>  		threshs14;
static ThresholdsActivation<L15_TMEM, L15_PE, L15_API, ap_int<16>, ap_uint<L15_API>>  		threshs15;
static ThresholdsActivation<L16_TMEM, L16_PE, L16_API, ap_int<16>, ap_uint<L16_API>>  		threshs16;

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
	  threshs11.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 24:
	  weights12.m_weights[targetMem][targetInd] = val;
	  break;
	case 25:
	  threshs12.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 26:
	  weights13.m_weights[targetMem][targetInd] = val;
	  break;
	case 27:
	  threshs13.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 28:
	  weights14.m_weights[targetMem][targetInd] = val;
	  break;
	case 29:
	  threshs14.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 30:
	  weights15.m_weights[targetMem][targetInd] = val;
	  break;
	case 31:
	  threshs15.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 32:
	  weights16.m_weights[targetMem][targetInd] = val;
	  break;
	case 33:
	  threshs16.m_thresholds[targetMem][targetInd][targetThresh] = val;
	  break;
	case 34:
	  weights17.m_weights[targetMem][targetInd] = val;
	  break;
	case 35:
	  //do nothing
	  break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64>* out, const unsigned int numReps) {
#pragma HLS DATAFLOW

	stream<ap_uint<64> > instream("DoCompute.instream");
	stream<ap_uint<8*IMG_CH> > instream_bitw("DoCompute.instream_bitw");
#pragma HLS STREAM variable=instream_bitw depth=128

	// 0th conv and 0th max pool
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
	
	// 0th inception 3x3 conv
	stream<ap_uint<L1_OFM_CH> > inception0conv3x3("DoCompute.inception0conv3x3");
#pragma HLS STREAM variable=inception0conv3x3 depth=128

	// 0th inception 5x5 conv
	stream<ap_uint<L2_OFM_CH> > inception0conv5x5("DoCompute.inception0conv5x5");
#pragma HLS STREAM variable=inception0conv5x5 depth=128

	// 0th inception 7x7 conv
	stream<ap_uint<L3_OFM_CH> > inception0conv7x7("DoCompute.inception0conv7x7");
#pragma HLS STREAM variable=inception0conv7x7 depth=128

    // 0th inception concat stream and pool stream
	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>> inception0concatstream("DoCompute.inception0concatstream");
#pragma HLS STREAM variable=inception0concatstream depth=128
	stream<ap_uint<L1_OFM_CH+L2_OFM_CH+L3_OFM_CH> > inception0poolstream("DoCompute.inception0poolstream");
#pragma HLS STREAM variable=inception0poolstream depth=128

    // 1st conv and max pool
	stream<ap_uint<L4_OFM_CH> > convstream1("DoCompute.convstream1");
	stream<ap_uint<L4_OFM_CH> > poolstream1("DoCompute.poolstream1");
#pragma HLS STREAM variable=poolstream1 depth=128

	 // 2nd conv and max pool
	stream<ap_uint<L5_OFM_CH> > convstream2("DoCompute.convstream2");
	stream<ap_uint<L5_OFM_CH> > poolstream2("DoCompute.poolstream2");
#pragma HLS STREAM variable=poolstream2 depth=128

		// cloning streams
	stream<ap_uint<L5_OFM_CH>> poolstream2copy1("DoCompute.poolstream2copy1");
#pragma HLS STREAM variable=poolstream2copy1 depth=128
	stream<ap_uint<L5_OFM_CH>> poolstream2copy2("DoCompute.poolstream2copy2");
#pragma HLS STREAM variable=poolstream2copy2 depth=128
	stream<ap_uint<L5_OFM_CH>> poolstream2copy3("DoCompute.poolstream2copy3");
#pragma HLS STREAM variable=poolstream2copy3 depth=128


	// 1st inception 3x3 conv
	stream<ap_uint<L6_OFM_CH> > inception1conv3x3("DoCompute.inception1conv3x3");
#pragma HLS STREAM variable=inception1conv3x3 depth=128

	// 1st inception 5x5 conv
	stream<ap_uint<L7_OFM_CH> > inception1conv5x5("DoCompute.inception1conv5x5");
#pragma HLS STREAM variable=inception1conv5x5 depth=128

	// 1st inception 7x7 conv
	stream<ap_uint<L8_OFM_CH> > inception0conv7x7("DoCompute.inception1conv7x7");
#pragma HLS STREAM variable=inception1conv7x7 depth=128

    // 1st inception concat stream and pool stream
	stream<ap_uint<L6_OFM_CH+L7_OFM_CH+L8_OFM_CH>> inception1concatstream("DoCompute.inception1concatstream");
#pragma HLS STREAM variable=inception1concatstream depth=128

	stream<ap_uint<L6_OFM_CH+L7_OFM_CH+L8_OFM_CH> > inception1poolstream("DoCompute.inception1poolstream");
#pragma HLS STREAM variable=inception1poolstream depth=128


    // 3rd conv and max pool
	stream<ap_uint<L9_OFM_CH> > convstream3("DoCompute.convstream3");
	stream<ap_uint<L9_OFM_CH> > poolstream3("DoCompute.poolstream3");
#pragma HLS STREAM variable=poolstream3 depth=128

	 // 4th conv and max pool
	stream<ap_uint<L10_OFM_CH> > convstream4("DoCompute.convstream4");
	stream<ap_uint<L10_OFM_CH> > poolstream4("DoCompute.poolstream4");
#pragma HLS STREAM variable=poolstream4 depth=128


    // cloning streams
	stream<ap_uint<L10_OFM_CH>> poolstream4copy1("DoCompute.poolstream4copy1");
#pragma HLS STREAM variable=poolstream4copy1 depth=128
	stream<ap_uint<L10_OFM_CH>> poolstream4copy2("DoCompute.poolstream4copy2");
#pragma HLS STREAM variable=poolstream4copy2 depth=128
	stream<ap_uint<L10_OFM_CH>> poolstream4copy3("DoCompute.poolstream4copy3");
#pragma HLS STREAM variable=poolstream4copy3 depth=128


	// 2nd inception 3x3 conv
	stream<ap_uint<L11_OFM_CH> > inception2conv3x3("DoCompute.inception2conv3x3");
#pragma HLS STREAM variable=inception2conv3x3 depth=128

	// 2nd inception 5x5 conv
	stream<ap_uint<L12_OFM_CH> > inception2conv5x5("DoCompute.inception2conv5x5");
#pragma HLS STREAM variable=inception2conv5x5 depth=128

	// 2nd inception 7x7 conv
	stream<ap_uint<L13_OFM_CH> > inception2conv7x7("DoCompute.inception2conv7x7");
#pragma HLS STREAM variable=inception2conv7x7 depth=128

    // 1st inception concat stream and pool stream
	stream<ap_uint<L11_OFM_CH+L12_OFM_CH+L13_OFM_CH>> inception2concatstream("DoCompute.inception2concatstream");
#pragma HLS STREAM variable=inception2concatstream depth=128

	stream<ap_uint<L11_OFM_CH+L12_OFM_CH+L13_OFM_CH> > inception2poolstream("DoCompute.inception2poolstream");
#pragma HLS STREAM variable=inception2poolstream depth=128


    // 5th conv and max pool
	stream<ap_uint<L14_OFM_CH> > convstream5("DoCompute.convstream5");
	stream<ap_uint<L14_OFM_CH> > poolstream5("DoCompute.poolstream5");
#pragma HLS STREAM variable=poolstream5 depth=128

	 // 6th conv and max pool
	stream<ap_uint<L15_OFM_CH> > convstream6("DoCompute.convstream6");
	stream<ap_uint<L15_OFM_CH> > poolstream6("DoCompute.poolstream6");
#pragma HLS STREAM variable=poolstream6 depth=128


	stream<ap_uint<64> > fcstream1("DoCompute.fcstream1");
#pragma HLS STREAM variable=fcstream1 depth=128

	stream<ap_uint<64> > memOutStrm("DoCompute.memOutStrm");

	const unsigned int inBits = IMG_DIM*IMG_DIM*IMG_CH*8;
	const unsigned int outBits = L17_MH*16;

	Mem2Stream_Batch<64, inBits/8> (in, instream, numReps);
	StreamingDataWidthConverter_Batch<64, 8, (inBits) / 64> (instream, instream_bitw, numReps);

	// convolutional layers

	// 0th conv and pool
	ConvLayerValid_Batch<L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, L0_SIMD, L0_PE, Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Identity, Recast<Binary>>(instream_bitw, convstream0, weights0, threshs0, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L0_OFM_DIM, 2, L0_OFM_CH>(convstream0, poolstream0, numReps);

	// cloning one input stream into 3
	CloneStream_Batch<L0_OFM_CH, L1_IFM_DIM>(poolstream0, poolstream0copy1, poolstream0copy2, poolstream0copy3, numReps);
	
	/* inception 0 */
	// 3x3 conv block
	ConvLayerValid_Batch<L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, L1_SIMD, L1_PE, Recast<XnorMul>>(poolstream0copy1, inception0conv3x3, weights1, threshs1, numReps, ap_resource_lut());
	// 5x5 conv block
	ConvLayerSame_Batch<L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, L2_SIMD, L2_PE, Recast<XnorMul>>(poolstream0copy2, inception0conv5x5, weights2, threshs2, numReps, ap_resource_lut());
	// 7x7 conv block
	ConvLayerSame_Batch<L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, L3_SIMD, L3_PE, Recast<XnorMul>>(poolstream0copy3, inception0conv7x7, weights3, threshs3, numReps, ap_resource_lut());
	//concatenating  inception0conv3x3, inception0conv5x5 and inception0conv7x7
	ConcatStream_Batch<L3_OFM_CH, L3_OFM_DIM>(inception0conv3x3, inception0conv5x5, inception0conv7x7, inception0concatstream, numReps);

	StreamingMaxPoolEven_Batch<L3_OFM_DIM, 2, L1_OFM_CH+L2_OFM_CH+L3_OFM_CH>(inception0concatstream, inception0poolstream, numReps);

    
    // 1st conv and pool
	ConvLayerSame_Batch<L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, L4_SIMD, L4_PE, Recast<XnorMul>>(inception0poolstream, convstream1, weights4, threshs4, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L4_OFM_DIM, 2, L4_OFM_CH>(convstream1, poolstream1, numReps);

     // 2nd conv and pool
	ConvLayerSame_Batch<L5_K, L5_IFM_CH, L5_IFM_DIM, L5_OFM_CH, L5_OFM_DIM, L5_SIMD, L5_PE, Recast<XnorMul>>(poolstream1, convstream2, weights5, threshs5, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L5_OFM_DIM, 2, L5_OFM_CH>(convstream2, poolstream2, numReps);

	// cloning one input stream into 3
	CloneStream_Batch<L5_OFM_CH, L6_IFM_DIM>(poolstream2, poolstream2copy1, poolstream2copy2, poolstream2copy3, numReps);
	
	/* inception 1 */
	// 3x3 conv block
	ConvLayerValid_Batch<L6_K, L6_IFM_CH, L6_IFM_DIM, L6_OFM_CH, L6_OFM_DIM, L6_SIMD, L6_PE, Recast<XnorMul>>(poolstream2copy1, inception1conv3x3, weights6, threshs6, numReps, ap_resource_lut());
	// 5x5 conv block
	ConvLayerSame_Batch<L7_K, L7_IFM_CH, L7_IFM_DIM, L7_OFM_CH, L7_OFM_DIM, L7_SIMD, L7_PE, Recast<XnorMul>>(poolstream2copy2, inception1conv5x5, weights7, threshs7, numReps, ap_resource_lut());
	// 7x7 conv block
	ConvLayerSame_Batch<L8_K, L8_IFM_CH, L8_IFM_DIM, L8_OFM_CH, L8_OFM_DIM, L8_SIMD, L8_PE, Recast<XnorMul>>(poolstream2copy3, inception1conv7x7, weights8, threshs8, numReps, ap_resource_lut());
	//concatenating  inception1conv3x3, inception1conv5x5 and inception1conv7x7
	ConcatStream_Batch<L8_OFM_CH, L8_OFM_DIM>(inception1conv3x3, inception1conv5x5, inception1conv7x7, inception1concatstream, numReps);

	StreamingMaxPoolEven_Batch<L8_OFM_DIM, 2, L6_OFM_CH+L7_OFM_CH+L8_OFM_CH>(inception1concatstream, inception1poolstream, numReps);

    
    // 3rd conv and pool
	ConvLayerSame_Batch<L9_K, L9_IFM_CH, L9_IFM_DIM, L9_OFM_CH, L9_OFM_DIM, L9_SIMD, L9_PE, Recast<XnorMul>>(inception1poolstream, convstream3, weights9, threshs9, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L9_OFM_DIM, 2, L9_OFM_CH>(convstream3, poolstream3, numReps);

     // 4th conv and pool
	ConvLayerSame_Batch<L10_K, L10_IFM_CH, L10_IFM_DIM, L10_OFM_CH, L10_OFM_DIM, L10_SIMD, L10_PE, Recast<XnorMul>>(poolstream3, convstream4, weights10, threshs10, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L10_OFM_DIM, 2, L10_OFM_CH>(convstream4, poolstream4, numReps);

    // cloning one input stream into 3
	CloneStream_Batch<L10_OFM_CH, L11_IFM_DIM>(poolstream4, poolstream4copy1, poolstream4copy2, poolstream4copy3, numReps);
	
	/* inception 2 */
	// 3x3 conv block
	ConvLayerValid_Batch<L11_K, L11_IFM_CH, L11_IFM_DIM, L11_OFM_CH, L11_OFM_DIM, L11_SIMD, L11_PE, Recast<XnorMul>>(poolstream4copy1, inception2conv3x3, weights11, threshs11, numReps, ap_resource_lut());
	// 5x5 conv block
	ConvLayerSame_Batch<L12_K, L12_IFM_CH, L12_IFM_DIM, L12_OFM_CH, L12_OFM_DIM, L12_SIMD, L12_PE, Recast<XnorMul>>(poolstream4copy2, inception2conv5x5, weights12, threshs12, numReps, ap_resource_lut());
	// 7x7 conv block
	ConvLayerSame_Batch<L13_K, L13_IFM_CH, L13_IFM_DIM, L13_OFM_CH, L13_OFM_DIM, L13_SIMD, L13_PE, Recast<XnorMul>>(poolstream4copy3, inception2conv7x7, weights13, threshs13, numReps, ap_resource_lut());
	//concatenating  inception2conv3x3, inception2conv5x5 and inception2conv7x7
	ConcatStream_Batch<L13_OFM_CH, L13_OFM_DIM>(inception2conv3x3, inception2conv5x5, inception2conv7x7, inception2concatstream, numReps);

	StreamingMaxPoolEven_Batch<L13_OFM_DIM, 2, L11_OFM_CH+L12_OFM_CH+L13_OFM_CH>(inception2concatstream, inception2poolstream, numReps);

    
    // 5th conv and pool
	ConvLayerSame_Batch<L14_K, L14_IFM_CH, L14_IFM_DIM, L14_OFM_CH, L14_OFM_DIM, L14_SIMD, L14_PE, Recast<XnorMul>>(inception2poolstream, convstream5, weights14, threshs14, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L4_OFM_DIM, 2, L4_OFM_CH>(convstream5, poolstream5, numReps);

     // 6th conv and pool
	ConvLayerSame_Batch<L15_K, L15_IFM_CH, L15_IFM_DIM, L15_OFM_CH, L15_OFM_DIM, L15_SIMD, L15_PE, Recast<XnorMul>>(poolstream5, convstream6, weights15, threshs15, numReps, ap_resource_lut());
	StreamingMaxPoolEven_Batch<L15_OFM_DIM, 2, L15_OFM_CH>(convstream6, poolstream6, numReps);


    
	// fully connected layers
	WidthAdjustedOutputStream<L17_PE, 64, L17_MH / L17_PE>  wa_out(memOutStrm, numReps);
	StreamingFCLayer_Batch<L16_MW, L16_MH, L16_SIMD, L16_PE, Recast<XnorMul>>
	(poolstream6, fcstream1,  weights16, threshs16, numReps, ap_resource_lut());
	StreamingFCLayer_Batch<L17_MW, L17_MH, L17_SIMD, L17_PE, Recast<XnorMul>, Slice<ap_uint<16> >>
	(fcstream1, static_cast<hls::stream<ap_uint<L17_PE>>&>(wa_out), weights17, PassThroughActivation<ap_uint<16>>(), numReps, ap_resource_lut());

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
#pragma HLS ARRAY_PARTITION variable=threshs11.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs11.m_thresholds complete dim=3

	#pragma HLS ARRAY_PARTITION variable=weights12.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs12.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs12.m_thresholds complete dim=3

	#pragma HLS ARRAY_PARTITION variable=weights13.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs13.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs13.m_thresholds complete dim=3

	#pragma HLS ARRAY_PARTITION variable=weights14.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs14.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs14.m_thresholds complete dim=3

	#pragma HLS ARRAY_PARTITION variable=weights15.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs15.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs15.m_thresholds complete dim=3

	#pragma HLS ARRAY_PARTITION variable=weights16.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs16.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs16.m_thresholds complete dim=3

	#pragma HLS ARRAY_PARTITION variable=weights17.m_weights complete dim=1


  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }
}
