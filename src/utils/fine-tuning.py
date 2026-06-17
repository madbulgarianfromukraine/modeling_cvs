# %% [code]
import torch
import torch.nn as nn

def initial_freeze_unfreeze(model):
    for param in model.parameters():
        param.requires_grad = False

    for cls_arg in model.classifier:
        for param in cls_arg.parameters():
            param.requires_grad = True


def second_unfreeze(model):
    for feature_arg in model.features:
        for param in feature_arg.parameters():
            param.requires_grad = True


def reset_head(model, input_features, output_features):
    for layer in model.classifier:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    fc_last_in_features = model.classifier[-1].in_features
    model.classifier[1] = nn.Linear(in_features=input_features, out_features=fc_last_in_features)
    model.classifier[-1] = nn.Linear(in_features=fc_last_in_features, out_features=output_features) # exactly so much classes are in Enrico

def reset_head_last_fc(model, output_features):
    for layer in model.classifier:
        if hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    fc_last_in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=fc_last_in_features, out_features=output_features)