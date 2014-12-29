
TARGETS = lib/libcutil_x86_64.a canny 

all: $(TARGETS)

canny: canny.cu
	nvcc -arch=sm_13 -O -Icommon/inc canny.cu -Llib -lcutil_x86_64 -o canny

testDiffs: testDiffs.cu
	nvcc -arch=sm_13 -O -Icommon/inc testDiffs.cu -Llib -lcutil_x86_64 -o testDiffs

lib/libcutil_x86_64.a: 
	make -C common

clean:
	make -C common clean
	rm -f $(TARGETS) 
	rm -f reference.pgm lenaOut.pgm
