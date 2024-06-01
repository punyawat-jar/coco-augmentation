#!/bin/bash

roboflow login;

roboflow import -w image-process-mahidol -p buu-lspine-ap-aug -c 30 -n AP_ClassMix2 -r 20 ./AP/images;
