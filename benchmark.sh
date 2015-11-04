#!/bin/bash

function benchmark {
  
  echo
  echo "***********************************"
  echo  $PROMPT
  echo "***********************************"
  echo

  for f in $FILES; do \
      rm -f $f.gz
      echo current file: $f
      sizebefore=$(echo "scale=2;$(stat -c "%s" $f)/1000000" | bc)
      t=$(/usr/bin/time -f "%e" $COMMAND $f 2>&1 >/dev/null )
      sizeafter=$(echo "scale=2;$(stat -c "%s" $f.gz)/1000000" | bc)
      ratio=$(echo "scale=2;$sizebefore/$sizeafter" | bc)
      echo size before:$sizebefore MB
      echo size after: $sizeafter MB
      echo time: $t seconds
      echo ratio: $ratio
      speed=$(echo "scale=2;""$sizebefore / $t" | bc)
      echo speed: $speed MB/s
      rm -f $f.gz
      echo
  done
}

rm -f corpus/*.gz

PROMPT="Multi threaded pigz -9 benchmarks."
FILES='corpus/combined corpus/*.big'
COMMAND='pigz/pigz -f -k -9'
benchmark

PROMPT="Single threaded pigz -9 benchmarks."
FILES='corpus/combined corpus/*.big'
COMMAND='pigz/pigzn -f -k -9'
benchmark

PROMPT="Multi threaded pigz -11 benchmarks."
FILES='corpus/combined corpus/*.orig'
COMMAND='pigz/pigz -f -k -11'
benchmark


PROMPT="Single threaded pigz -11 benchmarks."
FILES='corpus/combined corpus/*.orig'
COMMAND='pigz/pigzn -f -k -11'
benchmark

