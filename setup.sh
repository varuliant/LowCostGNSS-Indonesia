#!/bin/bash
# Kompilasi source code C rnx2rtkp untuk Linux di server cloud
gcc -Wall -O3 -I./rtklib_src ./rtklib_src/rnx2rtkp.c ./rtklib_src/rtkcmn.c ./rtklib_src/rinex.c -o rnx2rtkp -lm
chmod +x rnx2rtkp