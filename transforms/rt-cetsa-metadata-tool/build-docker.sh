#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rt-cetsa-metadata-tool:"${version}"
