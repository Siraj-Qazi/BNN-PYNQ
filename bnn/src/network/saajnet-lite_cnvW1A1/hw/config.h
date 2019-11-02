/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =    28  IFM_CH =     1
 *      OFM  =    28  OFM_CH =     8
 *     SIMD  =     1    PE   =     1
 *     WMEM  =    72   TMEM  =     8
 *     #Ops  = 112896   Ext Latency  = 56448
**/

#define L0_K 3
#define L0_IFM_CH 1
#define L0_IFM_DIM 28
#define L0_OFM_CH 8
#define L0_OFM_DIM 28
#define L0_SIMD 1
#define L0_PE 1
#define L0_WMEM 72
#define L0_TMEM 8
#define L0_WPI 1
#define L0_API 1
#define L0_WPF 0
#define L0_APF 0

/**
 * Convolutional Layer L1:
 *      IFM  =    14  IFM_CH =     8
 *      OFM  =    14  OFM_CH =    16
 *     SIMD  =     1    PE   =     1
 *     WMEM  =   128   TMEM  =    16
 *     #Ops  = 50176   Ext Latency  = 25088
**/

#define L1_K 1
#define L1_IFM_CH 8
#define L1_IFM_DIM 14
#define L1_OFM_CH 16
#define L1_OFM_DIM 14
#define L1_SIMD 1
#define L1_PE 1
#define L1_WMEM 128
#define L1_TMEM 16
#define L1_WPI 1
#define L1_API 1
#define L1_WPF 0
#define L1_APF 0

/**
 * Convolutional Layer L2:
 *      IFM  =    14  IFM_CH =     8
 *      OFM  =    14  OFM_CH =    16
 *     SIMD  =     1    PE   =     4
 *     WMEM  =   288   TMEM  =     4
 *     #Ops  = 451584   Ext Latency  = 56448
**/

#define L2_K 3
#define L2_IFM_CH 8
#define L2_IFM_DIM 14
#define L2_OFM_CH 16
#define L2_OFM_DIM 14
#define L2_SIMD 1
#define L2_PE 4
#define L2_WMEM 288
#define L2_TMEM 4
#define L2_WPI 1
#define L2_API 1
#define L2_WPF 0
#define L2_APF 0

/**
 * Convolutional Layer L3:
 *      IFM  =    14  IFM_CH =     8
 *      OFM  =    14  OFM_CH =    16
 *     SIMD  =     1    PE   =     4
 *     WMEM  =   800   TMEM  =     4
 *     #Ops  = 1254400   Ext Latency  = 156800
**/

#define L3_K 5
#define L3_IFM_CH 8
#define L3_IFM_DIM 14
#define L3_OFM_CH 16
#define L3_OFM_DIM 14
#define L3_SIMD 1
#define L3_PE 4
#define L3_WMEM 800
#define L3_TMEM 4
#define L3_WPI 1
#define L3_API 1
#define L3_WPF 0
#define L3_APF 0

/**
 * Convolutional Layer L4:
 *      IFM  =     7  IFM_CH =    48
 *      OFM  =     7  OFM_CH =    32
 *     SIMD  =     1    PE   =     1
 *     WMEM  =  1536   TMEM  =    32
 *     #Ops  = 150528   Ext Latency  = 75264
**/

#define L4_K 1
#define L4_IFM_CH 48
#define L4_IFM_DIM 7
#define L4_OFM_CH 32
#define L4_OFM_DIM 7
#define L4_SIMD 1
#define L4_PE 1
#define L4_WMEM 1536
#define L4_TMEM 32
#define L4_WPI 1
#define L4_API 1
#define L4_WPF 0
#define L4_APF 0

/**
 * Convolutional Layer L5:
 *      IFM  =     7  IFM_CH =    48
 *      OFM  =     7  OFM_CH =    32
 *     SIMD  =     4    PE   =     4
 *     WMEM  =   864   TMEM  =     8
 *     #Ops  = 1354752   Ext Latency  = 42336
**/

#define L5_K 3
#define L5_IFM_CH 48
#define L5_IFM_DIM 7
#define L5_OFM_CH 32
#define L5_OFM_DIM 7
#define L5_SIMD 4
#define L5_PE 4
#define L5_WMEM 864
#define L5_TMEM 8
#define L5_WPI 1
#define L5_API 1
#define L5_WPF 0
#define L5_APF 0

/**
 * Convolutional Layer L6:
 *      IFM  =     7  IFM_CH =    48
 *      OFM  =     7  OFM_CH =    32
 *     SIMD  =     4    PE   =     4
 *     WMEM  =  2400   TMEM  =     8
 *     #Ops  = 3763200   Ext Latency  = 117600
**/

#define L6_K 5
#define L6_IFM_CH 48
#define L6_IFM_DIM 7
#define L6_OFM_CH 32
#define L6_OFM_DIM 7
#define L6_SIMD 4
#define L6_PE 4
#define L6_WMEM 2400
#define L6_TMEM 8
#define L6_WPI 1
#define L6_API 1
#define L6_WPF 0
#define L6_APF 0

/**
 * Convolutional Layer L7:
 *      IFM  =     3  IFM_CH =    96
 *      OFM  =     3  OFM_CH =    64
 *     SIMD  =     1    PE   =     4
 *     WMEM  =  1536   TMEM  =    16
 *     #Ops  = 110592   Ext Latency  = 13824
**/

#define L7_K 1
#define L7_IFM_CH 96
#define L7_IFM_DIM 3
#define L7_OFM_CH 64
#define L7_OFM_DIM 3
#define L7_SIMD 1
#define L7_PE 4
#define L7_WMEM 1536
#define L7_TMEM 16
#define L7_WPI 1
#define L7_API 1
#define L7_WPF 0
#define L7_APF 0

/**
 * Convolutional Layer L8:
 *      IFM  =     3  IFM_CH =    96
 *      OFM  =     3  OFM_CH =    64
 *     SIMD  =     8    PE   =     4
 *     WMEM  =  1728   TMEM  =    16
 *     #Ops  = 995328   Ext Latency  = 15552
**/

#define L8_K 3
#define L8_IFM_CH 96
#define L8_IFM_DIM 3
#define L8_OFM_CH 64
#define L8_OFM_DIM 3
#define L8_SIMD 8
#define L8_PE 4
#define L8_WMEM 1728
#define L8_TMEM 16
#define L8_WPI 1
#define L8_API 1
#define L8_WPF 0
#define L8_APF 0

/**
 * Convolutional Layer L9:
 *      IFM  =     3  IFM_CH =    96
 *      OFM  =     3  OFM_CH =    64
 *     SIMD  =    16    PE   =     4
 *     WMEM  =  2400   TMEM  =    16
 *     #Ops  = 2764800   Ext Latency  = 21600
**/

#define L9_K 5
#define L9_IFM_CH 96
#define L9_IFM_DIM 3
#define L9_OFM_CH 64
#define L9_OFM_DIM 3
#define L9_SIMD 16
#define L9_PE 4
#define L9_WMEM 2400
#define L9_TMEM 16
#define L9_WPI 1
#define L9_API 1
#define L9_WPF 0
#define L9_APF 0

/**
 * Fully-Connected Layer L10:
 *     MatW =   192 MatH =  1024
 *     SIMD =    16  PE  =    16
 *     WMEM =   768 TMEM =    64
 *     #Ops  = 393216   Ext Latency  =   768
**/

#define L10_SIMD 16
#define L10_PE 16
#define L10_WMEM 768
#define L10_TMEM 64
#define L10_MW 192
#define L10_MH 1024
#define L10_WPI 1
#define L10_API 1
#define L10_WPF 0
#define L10_APF 0

/**
 * Fully-Connected Layer L11:
 *     MatW =  1024 MatH =    64
 *     SIMD =     1  PE  =    32
 *     WMEM =  2048 TMEM =     2
 *     #Ops  = 131072   Ext Latency  =  2048
**/

#define L11_SIMD 1
#define L11_PE 32
#define L11_WMEM 2048
#define L11_TMEM 2
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

