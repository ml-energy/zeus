# Copyright (C) 2023 Jae-Won Chung <jwnchung@umich.edu>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bridge between weiaicunzai/pytorch-cifar100 and Zeus."""


all_models = [
    "vgg16",
    "vgg13",
    "vgg11",
    "vgg19",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
    "googlenet",
    "inceptionv3",
    "inceptionv4",
    "inceptionresnetv2",
    "xception",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "preactresnet18",
    "preactresnet34",
    "preactresnet50",
    "preactresnet101",
    "preactresnet152",
    "resnext50",
    "resnext101",
    "resnext152",
    "shufflenet",
    "shufflenetv2",
    "squeezenet",
    "mobilenet",
    "mobilenetv2",
    "nasnet",
    "attention56",
    "attention92",
    "seresnet18",
    "seresnet34",
    "seresnet50",
    "seresnet101",
    "seresnet152",
    "wideresnet",
    "stochasticdepth18",
    "stochasticdepth34",
    "stochasticdepth50",
    "stochasticdepth101",
]


def get_model(name: str):
    """Instantiate the designated DNN model."""
    if name == "vgg16":
        from models.vgg import vgg16_bn
        model = vgg16_bn()
    elif name == "vgg13":
        from models.vgg import vgg13_bn
        model = vgg13_bn()
    elif name == "vgg11":
        from models.vgg import vgg11_bn
        model = vgg11_bn()
    elif name == "vgg19":
        from models.vgg import vgg19_bn
        model = vgg19_bn()
    elif name == "densenet121":
        from models.densenet import densenet121
        model = densenet121()
    elif name == "densenet161":
        from models.densenet import densenet161
        model = densenet161()
    elif name == "densenet169":
        from models.densenet import densenet169
        model = densenet169()
    elif name == "densenet201":
        from models.densenet import densenet201
        model = densenet201()
    elif name == "googlenet":
        from models.googlenet import googlenet
        model = googlenet()
    elif name == "inceptionv3":
        from models.inceptionv3 import inceptionv3
        model = inceptionv3()
    elif name == "inceptionv4":
        from models.inceptionv4 import inceptionv4
        model = inceptionv4()
    elif name == "inceptionresnetv2":
        from models.inceptionv4 import inception_resnet_v2
        model = inception_resnet_v2()
    elif name == "xception":
        from models.xception import xception
        model = xception()
    elif name == "resnet18":
        from models.resnet import resnet18
        model = resnet18()
    elif name == "resnet34":
        from models.resnet import resnet34
        model = resnet34()
    elif name == "resnet50":
        from models.resnet import resnet50
        model = resnet50()
    elif name == "resnet101":
        from models.resnet import resnet101
        model = resnet101()
    elif name == "resnet152":
        from models.resnet import resnet152
        model = resnet152()
    elif name == "preactresnet18":
        from models.preactresnet import preactresnet18
        model = preactresnet18()
    elif name == "preactresnet34":
        from models.preactresnet import preactresnet34
        model = preactresnet34()
    elif name == "preactresnet50":
        from models.preactresnet import preactresnet50
        model = preactresnet50()
    elif name == "preactresnet101":
        from models.preactresnet import preactresnet101
        model = preactresnet101()
    elif name == "preactresnet152":
        from models.preactresnet import preactresnet152
        model = preactresnet152()
    elif name == "resnext50":
        from models.resnext import resnext50
        model = resnext50()
    elif name == "resnext101":
        from models.resnext import resnext101
        model = resnext101()
    elif name == "resnext152":
        from models.resnext import resnext152
        model = resnext152()
    elif name == "shufflenet":
        from models.shufflenet import shufflenet
        model = shufflenet()
    elif name == "shufflenetv2":
        from models.shufflenetv2 import shufflenetv2
        model = shufflenetv2()
    elif name == "squeezenet":
        from models.squeezenet import squeezenet
        model = squeezenet()
    elif name == "mobilenet":
        from models.mobilenet import mobilenet
        model = mobilenet()
    elif name == "mobilenetv2":
        from models.mobilenetv2 import mobilenetv2
        model = mobilenetv2()
    elif name == "nasnet":
        from models.nasnet import nasnet
        model = nasnet()
    elif name == "attention56":
        from models.attention import attention56
        model = attention56()
    elif name == "attention92":
        from models.attention import attention92
        model = attention92()
    elif name == "seresnet18":
        from models.senet import seresnet18
        model = seresnet18()
    elif name == "seresnet34":
        from models.senet import seresnet34
        model = seresnet34()
    elif name == "seresnet50":
        from models.senet import seresnet50
        model = seresnet50()
    elif name == "seresnet101":
        from models.senet import seresnet101
        model = seresnet101()
    elif name == "seresnet152":
        from models.senet import seresnet152
        model = seresnet152()
    elif name == "wideresnet":
        from models.wideresidual import wideresnet
        model = wideresnet()
    elif name == "stochasticdepth18":
        from models.stochasticdepth import stochastic_depth_resnet18
        model = stochastic_depth_resnet18()
    elif name == "stochasticdepth34":
        from models.stochasticdepth import stochastic_depth_resnet34
        model = stochastic_depth_resnet34()
    elif name == "stochasticdepth50":
        from models.stochasticdepth import stochastic_depth_resnet50
        model = stochastic_depth_resnet50()
    elif name == "stochasticdepth101":
        from models.stochasticdepth import stochastic_depth_resnet101
        model = stochastic_depth_resnet101()
    else:
        raise NotImplemented(f"Model {name} is not supported.")

    return model
