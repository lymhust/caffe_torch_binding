caffe = {}
local ffi = require 'ffi'

ffi.cdef[[
void init(void* handle[1], const char* param_file, const char* model_file, const char* phase_name);
void do_forward(void* handle[1], THFloatTensor* bottom, THFloatTensor* output);
void do_backward(void* handle[1], THFloatTensor* gradOutput, THFloatTensor* gradInput);
void reset(void* handle[1]);
void set_mode_cpu();
void set_mode_gpu();
void set_device(int device_id);
unsigned int get_blob_index(void* handle[1], const char *query_blob_name);
void get_blob_data(void* handle[1], unsigned int blob_id, THFloatTensor* output);
void read_mean(const char* mean_file_path, THFloatTensor* mean_tensor);
void reshape(void* handle[1], int bsize, int cnum, int h, int w);
]]

caffe.C = ffi.load(package.searchpath('libtcaffe', package.cpath))
