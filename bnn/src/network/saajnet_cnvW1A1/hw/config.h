/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    28  IFM_CH =     1
 *      OFM  =    28  OFM_CH =    16
 *     SIMD  =     1    PE   =     1
 *     WMEM  =   144   TMEM  =    16
 *     #Ops  = 225792   Ext Latency  = 112896
**/

#define L0_K 3
#define L0_IFM_CH 1
#define L0_IFM_DIM 28
#define L0_OFM_CH 16
#define L0_OFM_DIM 28
#define L0_SIMD 1
#define L0_PE 1
#define L0_WMEM 144
#define L0_TMEM 16
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    14  OFM_CH =    32
 *     SIMD  =     4    PE   =     4
 *     WMEM  =    32   TMEM  =     8
 *     #Ops  = 200704   Ext Latency  =  6272
**/

#define L1_K 1
#define L1_IFM_CH 16
#define L1_IFM_DIM 14
#define L1_OFM_CH 32
#define L1_OFM_DIM 14
#define L1_SIMD 4
#define L1_PE 4
#define L1_WMEM 32
#define L1_TMEM 8
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    14  OFM_CH =    32
 *     SIMD  =     4    PE   =     4
 *     WMEM  =   288   TMEM  =     8
 *     #Ops  = 1806336   Ext Latency  = 56448
**/

#define L2_K 3
#define L2_IFM_CH 16
#define L2_IFM_DIM 14
#define L2_OFM_CH 32
#define L2_OFM_DIM 14
#define L2_SIMD 4
#define L2_PE 4
#define L2_WMEM 288
#define L2_TMEM 8
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Convolutional Layer L3:
 *      IFM  =    14  IFM_CH =    16
 *      OFM  =    14  OFM_CH =    32
 *     SIMD  =     4    PE   =     4
 *     WMEM  =   800   TMEM  =     8
 *     #Ops  = 5017600   Ext Latency  = 156800
**/

#define L3_K 5
#define L3_IFM_CH 16
#define L3_IFM_DIM 14
#define L3_OFM_CH 32
#define L3_OFM_DIM 14
#define L3_SIMD 4
#define L3_PE 4
#define L3_WMEM 800
#define L3_TMEM 8
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

/**
 * Convolutional Layer L4:
 *      IFM  =     7  IFM_CH =    96
 *      OFM  =     7  OFM_CH =    64
 *     SIMD  =     4    PE   =     4
 *     WMEM  =   384   TMEM  =    16
 *     #Ops  = 602112   Ext Latency  = 18816
**/

#define L4_K 1
#define L4_IFM_CH 96
#define L4_IFM_DIM 7
#define L4_OFM_CH 64
#define L4_OFM_DIM 7
#define L4_SIMD 4
#define L4_PE 4
#define L4_WMEM 384
#define L4_TMEM 16
#define L4_WPI 1
#define L4_API 1
#define L4_WPF 0
#define L4_APF 0

/**
 * Convolutional Layer L5:
 *      IFM  =     7  IFM_CH =    96
 *      OFM  =     7  OFM_CH =    64
 *     SIMD  =     4    PE   =     4
 *     WMEM  =  3456   TMEM  =    16
 *     #Ops  = 5419008   Ext Latency  = 169344
**/

#define L5_K 3
#define L5_IFM_CH 96
#define L5_IFM_DIM 7
#define L5_OFM_CH 64
#define L5_OFM_DIM 7
#define L5_SIMD 4
#define L5_PE 4
#define L5_WMEM 3456
#define L5_TMEM 16
#define L5_WPI 1
#define L5_API 1
#define L5_WPF 0
#define L5_APF 0

/**
 * Convolutional Layer L6:
 *      IFM  =     7  IFM_CH =    96
 *      OFM  =     7  OFM_CH =    64
 *     SIMD  =     4    PE   =     4
 *     WMEM  =  9600   TMEM  =    16
 *     #Ops  = 15052800   Ext Latency  = 470400
**/

#define L6_K 5
#define L6_IFM_CH 96
#define L6_IFM_DIM 7
#define L6_OFM_CH 64
#define L6_OFM_DIM 7
#define L6_SIMD 4
#define L6_PE 4
#define L6_WMEM 9600
#define L6_TMEM 16
#define L6_WPI 1
#define L6_API 1
#define L6_WPF 0
#define L6_APF 0

/**
 * Convolutional Layer L7:
 *      IFM  =     3  IFM_CH =   192
 *      OFM  =     3  OFM_CH =   128
 *     SIMD  =     8    PE   =     8
 *     WMEM  =   384   TMEM  =    16
 *     #Ops  = 442368   Ext Latency  =  3456
**/

#define L7_K 1
#define L7_IFM_CH 192
#define L7_IFM_DIM 3
#define L7_OFM_CH 128
#define L7_OFM_DIM 3
#define L7_SIMD 8
#define L7_PE 8
#define L7_WMEM 384
#define L7_TMEM 16
#define L7_WPI 1
#define L7_API 1
#define L7_WPF 0
#define L7_APF 0

/**
 * Convolutional Layer L8:
 *      IFM  =     3  IFM_CH =   192
 *      OFM  =     3  OFM_CH =   128
 *     SIMD  =     8    PE   =     8
 *     WMEM  =  3456   TMEM  =    16
 *     #Ops  = 3981312   Ext Latency  = 31104
**/

#define L8_K 3
#define L8_IFM_CH 192
#define L8_IFM_DIM 3
#define L8_OFM_CH 128
#define L8_OFM_DIM 3
#define L8_SIMD 8
#define L8_PE 8
#define L8_WMEM 3456
#define L8_TMEM 16
#define L8_WPI 1
#define L8_API 1
#define L8_WPF 0
#define L8_APF 0

/**
 * Convolutional Layer L9:
 *      IFM  =     3  IFM_CH =   192
 *      OFM  =     3  OFM_CH =   128
 *     SIMD  =    16    PE   =    16
 *     WMEM  =  2400   TMEM  =     8
 *     #Ops  = 11059200   Ext Latency  = 21600
**/

#define L9_K 5
#define L9_IFM_CH 192
#define L9_IFM_DIM 3
#define L9_OFM_CH 128
#define L9_OFM_DIM 3
#define L9_SIMD 16
#define L9_PE 16
#define L9_WMEM 2400
#define L9_TMEM 8
#define L9_WPI 1
#define L9_API 1
#define L9_WPF 0
#define L9_APF 0

/**
 * Fully-Connected Layer L10:
 *     MatW =   384 MatH =  1024
 *     SIMD =    64  PE  =    32
 *     WMEM =   192 TMEM =    32
 *     #Ops  = 786432   Ext Latency  =   192
**/

#define L10_SIMD 64
#define L10_PE 32
#define L10_WMEM 192
#define L10_TMEM 32
#define L10_MW 384
#define L10_MH 1024
#define L10_WPI 1
#define L10_API 1
#define L10_WPF 0
#define L10_APF 0

/**
 * Fully-Connected Layer L11:
 *     MatW =  1024 MatH =    64
 *     SIMD =     1  PE  =    64
 *     WMEM =  1024 TMEM =     1
 *     #Ops  = 131072   Ext Latency  =  1024
**/

#define L11_SIMD 1
#define L11_PE 64
#define L11_WMEM 1024
#define L11_TMEM 1
#define L11_MW 1024
#define L11_MH 64
#define L11_WPI 1
#define L11_API 16
#define L11_WPF 0
#define L11_APF 0


#define LL_MH 64
#define IMG_DIM 28
#define IMG_CH 1
#define no_cl 10

#endif //__LAYER_CONFIG_H_

