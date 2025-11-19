#!/usr/bin/env -S bash -e

echo 'Building "dual-trits:latest"'
echo '============================'
docker build -t 'dual-trits:latest' .
echo 'Built "dual-trits:latest" successfully!'
echo '============================'
echo

echo 'main.cpp Output:'
echo '============================'
docker run --rm 'dual-trits:latest'
echo '============================'
