from collections import OrderedDict

DATASETS = OrderedDict(
    [
        ("sst2_gpt2_4shot", {"dataset": "SST-2", "model": "GPT-2 (4-shot)"}),
        ("sst2_gpt2", {"dataset": "SST-2", "model": "GPT-2 (0-shot)"}),
        ("sitw_plda", {"dataset": "SITW", "model": "PLDA"}),
        ("fvcaus_plda", {"dataset": "FVCAUS", "model": "PLDA"}),
        # OJO: No coinciden con paper TMLR:
        ("cifar100_repvgg_a2", {"dataset": "CIFAR-100", "model": "RepVGG-A2"}),
        ("cifar100_resnet-20", {"dataset": "CIFAR-100", "model": "ResNet-20"}),
        ("cifar100_vgg19_bn", {"dataset": "CIFAR-100", "model": "VGG19-BN"}),
        ####
        ("pneumoniamnist_resnet50", {"dataset": "Pneum", "model": "ResNet-50"}),
        ("adrenalmnist_resnet50", {"dataset": "Adrenal", "model": "ResNet-50"}),
        ("pathmnist_resnet50", {"dataset": "Path", "model": "ResNet-50"}),
        ("iemocap_wav2vec_pt", {"dataset": "IEMOCAP", "model": "Wav2Vec 2.0 (PT)"}),
        ("agnews_gpt2", {"dataset": "AGNews", "model": "GPT-2"}),
        ("cifar10_resnet-20", {"dataset": "CIFAR-10", "model": "ResNet-20"}),
        ("cifar10_vgg19_bn", {"dataset": "CIFAR-10", "model": "VGG19"}),
        ("cifar10_repvgg_a2", {"dataset": "CIFAR-10", "model": "RepVGG-A2"}),
    ]
)
