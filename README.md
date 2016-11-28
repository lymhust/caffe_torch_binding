torch-caffe-binding
===================

A short binding to use Caffe as a module in Torch7. Has the same functionality as MATLAB bindings.

You have to have installed and built Caffe, then do this:

```bash
CAFFE_DIR=/*path-to-caffe-root*/ luarocks make
```

Supported functions:
```lua
Net:forward(input)

Net:updateGradInput(input, gradOutput)

Net:getBlobIndx(query_blob_name)

Net:getBlobData(blob_id)

Net:readMean(mean_file_path)

Net:reshape(bnum, cnum, h, w)

Net:saveModel(weights_file)

Net:initGPUMemoryScope()

Net:reset()

Net:setModeCPU()

Net:setModeGPU()

Net:setDevice(device_id)
```

Examples:
```lua
require 'caffe'

net = caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'test')
input = torch.FloatTensor(10,3,227,227)
output = net:forward(input)

gradOutput = torch.FloatTensor(10,1000,1,1)
gradInput = net:backward(input, gradOutput)
```

User can also use it inside a network as nn.Module, for example:

```lua
require 'caffe'

model = nn.Sequential()
model:add(caffe.Net('deploy.prototxt', 'bvlc_alexnet.caffemodel', 'test'))
model:add(nn.Linear(1000,1))
```
