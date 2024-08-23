// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include "rknn_api.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "fp16/Float16.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include "inference.h"

using namespace rknpu2;

#define MAX_TEXT_LINE_LENGTH 1024

#define IMAGENET_CLASSES_FILE "./model/class.txt"
// #define MODEL_PATH "./model/MobileNetV3_Large.rknn"
// #define MODEL_PATH "./model/MobileNetV3_Small_bird_real_freeze_mmse.rknn"
// #define MODEL_PATH "./model/MobileNetV3_Small_bird_real_freeze_modified_mmse.rknn"

// #define MODEL_PATH "./model/MobileNetV3_Small_bird_real_non_freeze_mmse.rknn"
#define MODEL_PATH "./model/MobileNetV3_Small_bird_real_non_freeze_modified_mmse.rknn"
// #define MODEL_PATH "./model/MobileNetV3_Small_bird_real.rknn"

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

static void softmax(float *array, int size) {
    // Find the maximum value in the array
    float max_val = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max_val) {
            max_val = array[i];
        }
    }

    // Subtract the maximum value from each element to avoid overflow
    for (int i = 0; i < size; i++) {
        array[i] -= max_val;
    }

    // Compute the exponentials and sum
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        array[i] = expf(array[i]);
        sum += array[i];
    }

    // Normalize the array by dividing each element by the sum
    for (int i = 0; i < size; i++) {
        array[i] /= sum;
    }
}
typedef struct {
    float value;
    int index;
} element_t;

typedef struct {
    int cls;
    float score;
} mobilenet_result;

static void swap(element_t *a, element_t *b) {
    element_t temp = *a;
    *a = *b;
    *b = temp;
}

static int partition(element_t arr[], int low, int high) {
    float pivot = arr[high].value;
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j].value >= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }

    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}
static void quick_sort(element_t arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

static void get_topk_with_indices(float arr[], int size, int k, mobilenet_result *result) {

    // Create an array of elements, saving values ​​and index numbers
    element_t *elements = (element_t *)malloc(size * sizeof(element_t));
    for (int i = 0; i < size; i++) {
        elements[i].value = arr[i];
        elements[i].index = i;
    }

    // Quick sort an array of elements
    quick_sort(elements, 0, size - 1);

    // Get the top K maximum values ​​and their index numbers
    for (int i = 0; i < k; i++) {
        result[i].score = elements[i].value;
        result[i].cls = elements[i].index;
    }

    free(elements);
}

int count_lines(FILE* file)
{
    int count = 0;
    char ch;

    while(!feof(file))
    {
        ch = fgetc(file);
        if(ch == '\n')
        {
            count++;
        }
    }
    count += 1;

    rewind(file);
    return count;
}

char** read_lines_from_file(const char* filename, int* line_count)
{
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open the file.\n");
        return NULL;
    }

    int num_lines = count_lines(file);
    printf("num_lines=%d\n", num_lines);
    char** lines = (char**)malloc(num_lines * sizeof(char*));
    memset(lines, 0, num_lines * sizeof(char*));

    char buffer[MAX_TEXT_LINE_LENGTH];
    int line_index = 0;

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        buffer[strcspn(buffer, "\n")] = '\0';  // 移除换行符

        lines[line_index] = (char*)malloc(strlen(buffer) + 1);
        strcpy(lines[line_index], buffer);

        line_index++;
    }

    fclose(file);

    *line_count = num_lines;
    return lines;
}

static int rknn_GetTopN(float *pfProb, float *pfMaxProb, uint32_t *pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;
  uint32_t top_count = outputCount > topNum ? topNum : outputCount;

  for (i = 0; i < topNum; ++i)
  {
    pfMaxProb[i] = -FLT_MAX;
    pMaxClass[i] = -1;
  }

  for (j = 0; j < top_count; j++)
  {
    for (i = 0; i < outputCount; i++)
    {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4)))
      {
        continue;
      }

      float prob = pfProb[i];
      if (prob > *(pfMaxProb + j))
      {
        *(pfMaxProb + j) = prob;
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}

static int rknn_GetTopN_int8(int8_t *pProb, float scale, int zp, float *pfMaxProb, uint32_t *pMaxClass,
                             uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;
  uint32_t top_count = outputCount > topNum ? topNum : outputCount;

  for (i = 0; i < topNum; ++i)
  {
    pfMaxProb[i] = -FLT_MAX;
    pMaxClass[i] = -1;
  }

  for (j = 0; j < top_count; j++)
  {
    for (i = 0; i < outputCount; i++)
    {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4)))
      {
        continue;
      }

      float prob = (pProb[i] - zp) * scale;
      if (prob > *(pfMaxProb + j))
      {
        *(pfMaxProb + j) = prob;
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
  char dims[128] = {0};
  for (int i = 0; i < attr->n_dims; ++i)
  {
    int idx = strlen(dims);
    sprintf(&dims[idx], "%d%s", attr->dims[i], (i == attr->n_dims - 1) ? "" : ", ");
  }
  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, dims, attr->n_elems, attr->size, get_format_string(attr->fmt),
         get_type_string(attr->type), get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static void *load_file(const char *file_path, size_t *file_size)
{
  FILE *fp = fopen(file_path, "r");
  if (fp == NULL)
  {
    printf("failed to open file: %s\n", file_path);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  size_t size = (size_t)ftell(fp);
  fseek(fp, 0, SEEK_SET);

  void *file_data = malloc(size);
  if (file_data == NULL)
  {
    fclose(fp);
    printf("failed allocate file size: %zu\n", size);
    return NULL;
  }

  if (fread(file_data, 1, size, fp) != size)
  {
    fclose(fp);
    free(file_data);
    printf("failed to read file data!\n");
    return NULL;
  }

  fclose(fp);

  *file_size = size;

  return file_data;
}

static unsigned char *load_image(const char *image_path, rknn_tensor_attr *input_attr)
{
  int req_height = 0;
  int req_width = 0;
  int req_channel = 0;

  switch (input_attr->fmt)
  {
  case RKNN_TENSOR_NHWC:
    req_height = input_attr->dims[1];
    req_width = input_attr->dims[2];
    req_channel = input_attr->dims[3];
    break;
  case RKNN_TENSOR_NCHW:
    req_height = input_attr->dims[2];
    req_width = input_attr->dims[3];
    req_channel = input_attr->dims[1];
    break;
  default:
    printf("meet unsupported layout\n");
    return NULL;
  }

  int height = 0;
  int width = 0;
  int channel = 0;

  unsigned char *image_data = stbi_load(image_path, &width, &height, &channel, req_channel);
  if (image_data == NULL)
  {
    printf("load image failed!\n");
    return NULL;
  }

  if (width != req_width || height != req_height)
  {
    unsigned char *image_resized = (unsigned char *)STBI_MALLOC(req_width * req_height * req_channel);
    if (!image_resized)
    {
      printf("malloc image failed!\n");
      STBI_FREE(image_data);
      return NULL;
    }
    if (stbir_resize_uint8(image_data, width, height, 0, image_resized, req_width, req_height, 0, channel) != 1)
    {
      printf("resize image failed!\n");
      STBI_FREE(image_data);
      return NULL;
    }
    STBI_FREE(image_data);
    image_data = image_resized;
  }

  return image_data;
}

// 量化模型的npu输出结果为int8数据类型，后处理要按照int8数据类型处理
// 如下提供了int8排布的NC1HWC2转换成float的nchw转换代码
int NC1HWC2_int8_to_NCHW_float(const int8_t *src, float *dst, int *dims, int channel, int h, int w, int zp, float scale)
{
  int batch = dims[0];
  int C1 = dims[1];
  int C2 = dims[4];
  int hw_src = dims[2] * dims[3];
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++)
  {
    src = src + i * C1 * hw_src * C2;
    dst = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c)
    {
      int plane = c / C2;
      const int8_t *src_c = plane * hw_src * C2 + src;
      int offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w)
        {
          int cur_hw = cur_h * w + cur_w;
          dst[c * hw_dst + cur_h * w + cur_w] = (src_c[C2 * cur_hw + offset] - zp) * scale; // int8-->float
        }
    }
  }

  return 0;
}

// 量化模型的npu输出结果为fp16数据类型，后处理要按照fp16数据类型处理
// 如下提供了fp16排布的NC1HWC2转换成float的nchw转换代码
int NC1HWC2_fp16_to_NCHW_fp32(const float16* src, float* dst, int* dims, int channel, int h, int w, int zp, float scale)
{
  int batch  = dims[0];
  int C1     = dims[1];
  int C2     = dims[4];
  int hw_src = dims[2] * dims[3];
  int hw_dst = h * w;
  for (int i = 0; i < batch; i++) {
    const float16* src_b = src + i * C1 * hw_src * C2;
    float*         dst_b = dst + i * channel * hw_dst;
    for (int c = 0; c < channel; ++c) {
      int            plane  = c / C2;
      const float16* src_bc = plane * hw_src * C2 + src_b;
      int            offset = c % C2;
      for (int cur_h = 0; cur_h < h; ++cur_h)
        for (int cur_w = 0; cur_w < w; ++cur_w) {
          int cur_hw                 = cur_h * w + cur_w;
          dst_b[c * hw_dst + cur_hw] = src_bc[C2 * cur_hw + offset]; // float16-->float
        }
    }
  }

  return 0;
}



/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int inference3(char * input_data1, char * input_data2,char *input_data3, int * counter)
{
  int line_count;
  char** lines = read_lines_from_file(IMAGENET_CLASSES_FILE, &line_count);
  if (lines == NULL) {
      printf("read classes label file fail! path=%s\n", IMAGENET_CLASSES_FILE);
      return -1;
  }

  const char *model_path = MODEL_PATH;
  // char *input_path = argv[2];

  int loop_count = 1;
  // if (argc > 3)
  // {
  //   loop_count = atoi(argv[3]);
  // }

  rknn_context ctx = 0;

  // Load RKNN Model
#if 1
  // Init rknn from model path
  int ret = rknn_init(&ctx, (char *)model_path, 0, 0, NULL);
#else
  // Init rknn from model data
  size_t model_size;
  void *model_data = load_file(model_path, &model_size);
  if (model_data == NULL)
  {
    return -1;
  }
  int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
  free(model_data);
#endif
  if (ret < 0)
  {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get sdk and driver version
  rknn_sdk_version sdk_ver;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &sdk_ver, sizeof(sdk_ver));
  if (ret != RKNN_SUCC)
  {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("rknn_api/rknnrt version: %s, driver version: %s\n", sdk_ver.api_version, sdk_ver.drv_version);

  // Get Model Input Output Info
  rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC)
  {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }
  printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  // printf("input tensors:\n");
  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_input; i++)
  {
    input_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0)
    {
      printf("rknn_init error! ret=%d\n", ret);
      return -1;
    }
    // dump_tensor_attr(&input_attrs[i]);
  }

  // printf("output tensors:\n");
  rknn_tensor_attr output_attrs[io_num.n_output];
  memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++)
  {
    output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_NATIVE_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    // dump_tensor_attr(&output_attrs[i]);
  }

  // Get custom string
  // rknn_custom_string custom_string;
  // ret = rknn_query(ctx, RKNN_QUERY_CUSTOM_STRING, &custom_string, sizeof(custom_string));
  // if (ret != RKNN_SUCC)
  // {
  //   printf("rknn_query fail! ret=%d\n", ret);
  //   return -1;
  // }
  // printf("custom string: %s\n", custom_string.string);

  // unsigned char *input_data = NULL;
  rknn_tensor_type input_type = RKNN_TENSOR_UINT8;
  rknn_tensor_format input_layout = RKNN_TENSOR_NHWC;

  // Load image
  // input_data = load_image(input_path, &input_attrs[0]);

  if (!input_data1)
  {
    return -1;
  }

  // Create input tensor memory
  rknn_tensor_mem *input_mems[1];
  // default input type is int8 (normalize and quantize need compute in outside)
  // if set uint8, will fuse normalize and quantize to npu
  input_attrs[0].type = input_type;
  // default fmt is NHWC, npu only support NHWC in zero copy mode
  input_attrs[0].fmt = input_layout;

  input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride);

  // Copy input data to input tensor memory
  int width = input_attrs[0].dims[2];
  int stride = input_attrs[0].w_stride;

  // if (width == stride)
  // {
    memcpy(input_mems[0]->virt_addr, input_data1, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
  //   // printf("width==stride\n");
  // }


  // Create output tensor memory
  rknn_tensor_mem *output_mems[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i)
  {
    output_mems[i] = rknn_create_mem(ctx, output_attrs[i].size_with_stride);
  }

  // Set input tensor memory
  ret = rknn_set_io_mem(ctx, input_mems[0], &input_attrs[0]);
  if (ret < 0)
  {
    printf("rknn_set_io_mem fail! ret=%d\n", ret);
    return -1;
  }

  // Set output tensor memory
  for (uint32_t i = 0; i < io_num.n_output; ++i)
  {
    // set output memory and attribute
    ret = rknn_set_io_mem(ctx, output_mems[i], &output_attrs[i]);
    if (ret < 0)
    {
      printf("rknn_set_io_mem fail! ret=%d\n", ret);
      return -1;
    }
  }

  // Run
  printf("Begin perf ...\n");
  for (int i = 0; i < loop_count; ++i)
  {
    int64_t start_us = getCurrentTimeUs();
    ret = rknn_run(ctx, NULL);
    int64_t elapse_us = getCurrentTimeUs() - start_us;
    if (ret < 0)
    {
      printf("rknn run error %d\n", ret);
      return -1;
    }
    printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
  }

  printf("output origin tensors:\n");
  rknn_tensor_attr orig_output_attrs[io_num.n_output];
  memset(orig_output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
  for (uint32_t i = 0; i < io_num.n_output; i++)
  {
    orig_output_attrs[i].index = i;
    // query info
    ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(orig_output_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC)
    {
      printf("rknn_query fail! ret=%d\n", ret);
      return -1;
    }
    // dump_tensor_attr(&orig_output_attrs[i]);
  }

  float *output_mems_nchw[io_num.n_output];
  for (uint32_t i = 0; i < io_num.n_output; ++i)
  {
    int size = orig_output_attrs[i].size_with_stride * sizeof(float);
    output_mems_nchw[i] = (float *)malloc(size);
  }
  int8_t *src = (int8_t *)output_mems[0]->virt_addr;
  float *dst = output_mems_nchw[0];
  for (uint32_t i = 0; i < io_num.n_output; i++)
  {
    if (output_attrs[i].fmt == RKNN_TENSOR_NC1HWC2)
    {
      int channel = orig_output_attrs[i].dims[1];
      int h = orig_output_attrs[i].n_dims > 2 ? orig_output_attrs[i].dims[2] : 1;
      int w = orig_output_attrs[i].n_dims > 3 ? orig_output_attrs[i].dims[3] : 1;
      int zp = output_attrs[i].zp;
      float scale = output_attrs[i].scale;

      if (orig_output_attrs[i].type == RKNN_TENSOR_INT8) {
        NC1HWC2_int8_to_NCHW_float((int8_t *)output_mems[i]->virt_addr, (float *)output_mems_nchw[i], (int *)output_attrs[i].dims,
                                 channel, h, w, zp, scale);
      }
      else if (orig_output_attrs[i].type == RKNN_TENSOR_FLOAT16) {
        NC1HWC2_fp16_to_NCHW_fp32((float16*)output_mems[i]->virt_addr, (float*)output_mems_nchw[i],
                                  (int*)output_attrs[i].dims, channel, h, w, zp, scale);
      } else {
        printf("dtype: %s cannot convert!", get_type_string(orig_output_attrs[i].type));
      }
        // printf("类型为RKNN_TENSOR_NC1HWC2\n");

    }
    else
    {
      
      
      for (int index = 0; index < output_attrs[i].n_elems; index++)
      {
        dst[index] = ((float)src[index] - output_attrs[i].zp) * output_attrs[i].scale;
      }
        // printf("类型为RKNN_TENSOR_INT8\n");

    }
  }

  int topk = 2;
  mobilenet_result result[topk];
  int bird_cls = 0;
  // Post Process
  softmax(* output_mems_nchw, output_attrs[0].n_elems);
  get_topk_with_indices(* output_mems_nchw,output_attrs[0].n_elems, topk, result);


  for (int i = 0; i < topk; i++) {
        printf("[%d] score=%.6f class=%s\n", result[i].cls, result[i].score, lines[result[i].cls]);
    }
    if (result[0].cls ==bird_cls){
        // printf("[%d] score=%.6f class=%s\n", result[0].cls, result[0].score, lines[result[0].cls]);
        (*counter) ++;
        goto out;
    }



  // 若第一次为五鸟，则执行第二次推理
    memcpy(input_mems[0]->virt_addr, input_data2, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
    ret = rknn_run(ctx, NULL);
    for (int index = 0; index < output_attrs[0].n_elems; index++)
    {
      dst[index] = ((float)src[index] - output_attrs[0].zp) * output_attrs[0].scale;
    }
    softmax(* output_mems_nchw, output_attrs[0].n_elems);
    get_topk_with_indices(* output_mems_nchw,output_attrs[0].n_elems, topk, result);
    for (int i = 0; i < topk; i++) {
        printf("[%d] score=%.6f class=%s\n", result[i].cls, result[i].score, lines[result[i].cls]);
    }
    if (result[0].cls ==bird_cls){
        // printf("[%d] score=%.6f class=%s\n", result[0].cls, result[0].score, lines[result[0].cls]);
        (*counter) ++;
        goto out;

    }

  // 若第二次为五鸟，则执行第三次推理
    memcpy(input_mems[0]->virt_addr, input_data3, width * input_attrs[0].dims[1] * input_attrs[0].dims[3]);
    ret = rknn_run(ctx, NULL);
    for (int index = 0; index < output_attrs[0].n_elems; index++)
    {
      dst[index] = ((float)src[index] - output_attrs[0].zp) * output_attrs[0].scale;
    }
    softmax(* output_mems_nchw, output_attrs[0].n_elems);
    get_topk_with_indices(* output_mems_nchw,output_attrs[0].n_elems, topk, result);
    for (int i = 0; i < topk; i++) {
        printf("[%d] score=%.6f class=%s\n", result[i].cls, result[i].score, lines[result[i].cls]);
    }
    if (result[0].cls ==bird_cls){
        // printf("[%d] score=%.6f class=%s\n", result[0].cls, result[0].score, lines[result[0].cls]);
        (*counter) ++;
        goto out;

    }


out:
  // Destroy rknn memory
  rknn_destroy_mem(ctx, input_mems[0]);
  for (uint32_t i = 0; i < io_num.n_output; ++i)
  {
    rknn_destroy_mem(ctx, output_mems[i]);
    free(output_mems_nchw[i]);
  }

  // destroy
  rknn_destroy(ctx);

  // free(input_data);

  return 0;
}
