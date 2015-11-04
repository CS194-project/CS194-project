all:
	$(MAKE) -j -C zlib
	$(MAKE) -j -C pigz
	$(MAKE) -j -C pigz pigzn
debug:
	$(MAKE) -j -C zlib debug
	$(MAKE) -j -C pigz debug

clean:
	$(MAKE) -C zlib clean
	$(MAKE) -C pigz clean

test:
	$(MAKE) -C zlib test
	$(MAKE) -C pigz test

corpus:
	$(MAKE) -C corpus

.PHONY: all clean test corpus
