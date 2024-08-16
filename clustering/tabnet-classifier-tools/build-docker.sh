#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/pytorch-tabnet-tool:${version}
