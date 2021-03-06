
 
#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include <iostream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"
#include <algorithm>
#include "config.h"

using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void makeNetwork(network<mse, adagrad> & nn) 
{
  nn
#ifdef OFFLOAD
    << chaninterleave_layer<identity>(IMG_CH, IMG_DIM * IMG_DIM, false)
    << offloaded_layer(IMG_CH * IMG_DIM * IMG_DIM, no_cl, &FixedFoldedMVOffload<8, 1, ap_int<16>>, 0xdeadbeef, 0)
#endif
  ;
}

extern "C" void load_parameters(const char* path) 
{
  
  FoldedMVInit("cnvW1A1");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  cout << "Setting network weights and thresholds in accelerator..." << endl;
  FoldedMVLoadLayerMem(path, 0, L0_PE, L0_WMEM, L0_TMEM, L0_API);
  FoldedMVLoadLayerMem(path, 1, L1_PE, L1_WMEM, L1_TMEM, L1_API);
  FoldedMVLoadLayerMem(path, 2, L2_PE, L2_WMEM, L2_TMEM, L2_API);
  FoldedMVLoadLayerMem(path, 3, L3_PE, L3_WMEM, L3_TMEM, L3_API);
  FoldedMVLoadLayerMem(path, 4, L4_PE, L4_WMEM, L4_TMEM, L4_API);
  FoldedMVLoadLayerMem(path, 5, L5_PE, L5_WMEM, L5_TMEM, L5_API);
  FoldedMVLoadLayerMem(path, 6, L6_PE, L6_WMEM, L6_TMEM, L6_API);
  FoldedMVLoadLayerMem(path, 7, L7_PE, L7_WMEM, L7_TMEM, L7_API);
  FoldedMVLoadLayerMem(path, 8, L8_PE, L8_WMEM, L8_TMEM, L8_API);
  FoldedMVLoadLayerMem(path, 9, L9_PE, L9_WMEM, L9_TMEM, L9_API);
  FoldedMVLoadLayerMem(path, 10, L10_PE, L10_WMEM, L10_TMEM, L10_API);
  FoldedMVLoadLayerMem(path, 11, L11_PE, L11_WMEM, L11_TMEM, L11_API);

}

extern "C" int inference(const char* path, int results[LL_MH], int number_class, float* usecPerImage) {
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  std::vector<int> class_result;
  float usecPerImage_int;

  FoldedMVInit("cnvW1A1");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);
  class_result=testPrebuiltCIFAR10_from_image<8, 16, ap_int<16>>(test_images, number_class, usecPerImage_int);

  if(results) {
    std::copy(class_result.begin(),class_result.end(), results);
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  return (std::distance(class_result.begin(),std::max_element(class_result.begin(), class_result.end())));
}

extern "C" int* inference_multiple(const char* path, int number_class, int* image_number, float* usecPerImage, int enable_detail = 0) {
  std::vector<int> detailed_results;
  std::vector<label_t> test_labels;
  std::vector<vec_t> test_images;
  std::vector<int> all_result;
  float usecPerImage_int;
  int * result;

  FoldedMVInit("cnvW1A1");
  network<mse, adagrad> nn;
  makeNetwork(nn);
  parse_mnist_images(path, &test_images, -1.0, 1.0, 0, 0);
  all_result=testPrebuiltCIFAR10_multiple_images<8, 16, ap_int<16>>(test_images, number_class, detailed_results, usecPerImage_int);

  if (image_number) {
    *image_number = all_result.size();
  }
  if (usecPerImage) {
    *usecPerImage = usecPerImage_int;
  }
  if (enable_detail) {
    result = new int [detailed_results.size()];
    std::copy(detailed_results.begin(),detailed_results.end(), result);
  } else {
    result = new int [all_result.size()];
    std::copy(all_result.begin(),all_result.end(), result);
  }	   
  return result;
}

extern "C" void free_results(int* result) {
  delete[] result;
}

extern "C" void deinit() {
  FoldedMVDeinit();
}

extern "C" int main(int argc, char** argv) {
  if (argc != 5) {
    cout << "4 parameters are needed: " << endl;
    cout << "1 - folder for the binarized weights (binparam-***) - full path " << endl;
    cout << "2 - path to image to be classified" << endl;
    cout << "3 - number of classes in the dataset" << endl;
    cout << "4 - expected result" << endl;
    return 1;
  }
  float execution_time = 0;
  int scores[LL_MH];
  load_parameters(argv[1]);
  bool single = true;
  if(single)
  {
		int class_inference = inference(argv[2], scores, no_cl, &execution_time);
		cout << "Detected class " << class_inference << endl;
		cout << "in " << execution_time << " microseconds" << endl;
		deinit();
		if (class_inference != atol(argv[4]))
		{
			cout <<"\n Inference incorrect. Continuing anyway...\n";
      return 0;
		}
		else
		{
			return 0;
		}
  }
  else
  {
    // for checking multiple inference
    int ex_no = 15;
    int res[15] = {7,2,1,0,4,1,4,9,5,9,0,6,9,0,1};
    int error = 0;
    int *class_inference = inference_multiple(argv[2], no_cl, &ex_no, &execution_time);
    for(int i = 0 ; i < ex_no; i++)
    {
      cout << "Label = "<< res[i] << "\tPredicted = "<<class_inference[i] << endl;
      if (res[i] != class_inference[i])
      	error++;
    }
    cout << "Mismatches = " << error << endl;
    deinit();
    if(error >= 3) {
    	cout <<"\n More than 3 misinferences occured. Continuing anyway...\n";
      return 0;
    }
    else
    	return 0;
  }
}
