from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
from utils import *
from collections import defaultdict

# Make a list of dictionaries, where each dictionary represents each block of the cfg file
def parse_cfg(cfgfile):
    file = open(cfgfile)
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
            if block["type"] == "convolutional":
                block["batch_normalize"] = 0
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

# Create Convolutional, Upsample and Route layer modules from the list
def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            batch_normalize = int(x["batch_normalize"])
            bias = not batch_normalize
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if padding:
                pad = (kernel_size -1)//2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{}".format(index), conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{}".format(index), bn)
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky{}".format(index), activn)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsamp = nn.Upsample(scale_factor=2, mode="bilinear")
            module.add_module("upsamp{}".format(index), upsamp)
        
        # Route Layer possibly concatenates several previous layers or just passes single previous layer 
        # without any activation
        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0:
                start = start - index # Layer indices need to be negative, representing 
                                      # backward index of the layer to be routed
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route{}".format(index), route)
            if end < 0: # If end exists, i.e. two or more layers to be concatenated, feature map thickness is 
                        # addition of the two or more routed layers' thickness
                filters = output_filters[index + start] + output_filters[index + end]
            else:       # Else thickness is the same as of the layer routed
                filters = output_filters[index + start]
            
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        
        elif x["type"] == "maxpool":
            stride = int(x["stride"])
            size = int(x["size"])
            if stride != 1:
                maxpool = nn.MaxPool2d(size, stride)
            else:
                maxpool = MaxPoolStride1(size)
            
            module.add_module("maxpool_{}".format(index), maxpool)
        
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            inp_dim = int(net_info["height"])
            num_classes = int(x["classes"])
            detection = DetectionLayer(anchors, num_classes, inp_dim)
            module.add_module("detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)

# Create an empty layer to pass previous filters in route/shortcut layers without any convolution done
class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

# Create the YOLO layer to make detections on the feature map using anchors  
class DetectionLayer(nn.Module):
    def __init__(self, anchors, num_classes, inp_dim):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = inp_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss()  # Coordinate loss
        self.bce_loss = nn.BCELoss()  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                inp_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = 0
            if nProposals > 0:
                precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )

            return output

# Create Darknet from nn.ModuleList and design the forward pass of network and transform feature maps
# to be of the same dimensions, as feature maps at three different scales are passed through detection
# layer; final output has to be concatenated on single image of a fixed dimension
class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])
        self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]

    def forward(self, x, targets = None):
        modules = self.blocks[1:] # First block is information
        outputs = {} # Cache the outputs for the route/shortcut layers
        det_loss = [] # Cache the loss while training
        self.losses = defaultdict(float)
        write = 0 # No outpute layers at the beginning
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)
            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1) # concatenate
            
            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_] # add
            
            elif module_type == 'yolo':
                if targets is None: # Detection
                    x = self.module_list[i](x)
                    if not write:
                        detections = x
                        write = 1 # Now there is an output layer
                    else:
                        detections = torch.cat((detections, x), 1) # concatenate detections so far
                else: # Training
                    l, *losses = self.module_list[i][0](x, targets) 
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                    self.losses["recall"] /= 3
                    self.losses["precision"] /= 3
                    det_loss.append(l)

            outputs[i] = x
        
        return detections if targets is None else sum(det_loss)
    
    def load_weights(self, weightsfile):
        fp = open(weightsfile, "rb")
        header = np.fromfile(fp, dtype = np.int32, count =5) # First five integers are header
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        # Sequence of weights in weights file, only conv+batchnorm(bn) and conv layers have weights
        # bn biases > bn weights > bn running_mean > bn running_var > conv weights (conv+bn)
        # conv biases > conv weights (conv)

        weights = np.fromfile(fp, dtype = np.float32)

        ptr = 0 # Track position in the weights array
        # Populate weights and biases in respective modules of model
        for i, (block, module) in enumerate(zip(self.blocks[1:], self.module_list)):
            if block["type"] == "convolutional":
                conv = module[0] # Convolutional layer
                if block["batch_normalize"]: # Then bn biases and weights, if bn is true
                    bn = module[1] 
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr += num_bn_biases

                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases) # bn biases
                    bn.weight.data.copy_(bn_weights) # bn weights
                    bn.running_mean.copy_(bn_running_mean) # bn running_mean
                    bn.running_var.copy_(bn_running_var) # bn running_var
                else: # Then biases (conv biases), if bn is false
                    num_biases = conv.bias.numel()

                    conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
                    ptr += num_biases
                    
                    conv_biases = conv_biases.view_as(conv.bias.data) 
                    
                    conv.bias.data.copy_(conv_biases) # conv biases
                
                # Then conv weights, finally
                num_weights = conv.weight.numel()

                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]) # conv weights
                ptr += num_weights 
                
                conv_weights = conv_weights.view_as(conv.weight.data)
                
                conv.weight.data.copy_(conv_weights)

    def save_weights(self, path, cutoff=-1):
        fp = open(path, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)
        # Iterate through layers
        for i, (block, module) in enumerate(zip(self.blocks[1:cutoff], self.module_list[:cutoff])):
            if block["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, load bn first
                if block["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)
        fp.close()
