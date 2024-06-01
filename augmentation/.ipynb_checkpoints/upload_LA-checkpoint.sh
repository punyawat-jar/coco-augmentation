#!/bin/bash

roboflow login;

roboflow import -w image-process-mahidol-toon -p buu-lspine-la-aug-5s7w1 -c 30 -n LA_ClassMix2 -r 20 ./LA/images;
