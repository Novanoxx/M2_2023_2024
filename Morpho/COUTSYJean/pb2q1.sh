. ~/.pink.start

#Correct original image
seuil img/src/bloodcells.pgm 140 img/bin/bc.pgm
inverse img/bin/bc.pgm img/bin/bc_inv.pgm

areaclosing img/bin/bc_inv.pgm 8 200 img/bin/bc_ac.pgm
inverse img/bin/bc_ac.pgm img/bin/bc_hm_inv.pgm
pgm2GA img/bin/bc_hm_inv.pgm 1 img/bin/bc.ga
GA2khalimsky img/bin/bc.ga 0 img/result/pb2q1.pgm

#Display superposition
feh img/result/pb2q1.pgm
