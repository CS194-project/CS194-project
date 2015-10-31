#!/bin/bash
set -x
rm corpus/*.gz
for f in corpus/*; do { time pigz/pigz -k -9 $f; rm -f corpus/$f.gz; }; done
