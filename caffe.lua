local Net, parent = torch.class('caffe.Net', 'nn.Module')
local ffi = require 'ffi'
local C = caffe.C

function Net:__init(prototxt_name, binary_name, phase_name)
  assert(type(prototxt_name) == 'string')
  --assert(type(binary_name) == 'string')
  assert(type(phase_name) == 'string')
  parent.__init(self)
  self.handle = ffi.new'void*[1]'
  local old_handle = self.handle[1]
  C.init(self.handle, prototxt_name, binary_name, phase_name)
  if(self.handle[1] == old_handle) then
    print 'Unsuccessful init'
  end
  self.output = torch.Tensor()
  self.gradInput = torch.Tensor()
  self:float()
end

function Net:forward(input)
  assert(input:type() == 'torch.FloatTensor')
  C.do_forward(self.handle, input:cdata(), self.output:cdata())
  return self.output
end

function Net:updateGradInput(input, gradOutput)
  assert(input:type() == 'torch.FloatTensor')
  assert(gradOutput:type() == 'torch.FloatTensor')
  C.do_backward(self.handle, gradOutput:cdata(), self.gradInput:cdata())
  return self.gradInput
end

function Net:getBlobIndx(query_blob_name)
	assert(type(query_blob_name) == 'string')
	return C.get_blob_index(self.handle, ffi.cast('const char *', query_blob_name));
end

function Net:getBlobData(blob_id)
	assert(type(blob_id) == 'number')
	C.get_blob_data(self.handle, ffi.cast('unsigned int', blob_id), self.output:cdata());
	return self.output
end

function Net:readMean(mean_file_path)
	assert(type(mean_file_path) == 'string')
	local mean_tensor = torch.Tensor()
	C.read_mean(ffi.cast('const char *', mean_file_path), mean_tensor:cdata())
	return mean_tensor
end

function Net:reshape(bnum, cnum, h, w)
	assert(type(bnum) == 'number')
	assert(type(cnum) == 'number')
	assert(type(h) == 'number')
	assert(type(w) == 'number')
	C.reshape(self.handle, ffi.cast('int',bnum), ffi.cast('int',cnum), ffi.cast('int',h), ffi.cast('int',w))
end

function Net:reset()
  C.reset(self.handle)
end

function Net:setModeCPU()
  C.set_mode_cpu()
end

function Net:setModeGPU()
  C.set_mode_gpu()
end

function Net:setDevice(device_id)
  C.set_device(device_id)
end

