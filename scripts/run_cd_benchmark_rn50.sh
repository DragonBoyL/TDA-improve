#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python tda_runner3.py     --config configs \
                                                --wandb-log \
                                                --datasets dtd/oxford_flowers/ucf101 \
                                                --backbone RN50                                               