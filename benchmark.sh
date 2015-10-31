#!/bin/bash
set -x
rm -f corpus/*.gz
for f in corpus/*; do { time pigz/pigz -f -k -9 $f; rm -f corpus/$f.gz; }; done
