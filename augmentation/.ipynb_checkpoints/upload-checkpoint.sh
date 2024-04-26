#!/bin/bash

roboflow login;

roboflow import -w image-process-mahidol -p buu-lspine-la-aug -c 25 -n LA_Aug -r 20 ./AP/images;
roboflow import -w image-process-mahidol -p buu-lspine-ap-aug -c 25 -n AP_Aug -r 20 ./LA/images;
