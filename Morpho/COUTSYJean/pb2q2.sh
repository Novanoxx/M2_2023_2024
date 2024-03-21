. ~/.pink.start

#Correct original image
seuil img/src/bloodcells.pgm 140 img/bin/bc.pgm
inverse img/bin/bc.pgm img/bin/bc_inv.pgm

areaclosing img/bin/bc_inv.pgm 8 200 img/bin/bc_ac.pgm
inverse img/bin/bc_ac.pgm img/bin/pb2q1.pgm
pgm2GA img/bin/bc_hm_inv.pgm 1 img/bin/bc.ga
GA2khalimsky img/bin/bc.ga 0 img/bin/pb2q1.pgm

pgm2GA img/src/bloodcells.pgm 1 img/bin/bloodcells.ga
GA2khalimsky img/bin/bloodcells.ga 0 img/bin/bc_kha.pgm

GAwatershed img/bin/bc.ga img/bin/bc_ws.ga
frame img/bin/bc_kha.pgm 2 img/bin/bc_frame.pgm
inverse img/bin/pb2q1.pgm img/bin/bc_ws_inv.pgm
geodilat img/bin/bc_frame.pgm img/bin/bc_ws_inv.pgm 4 -1 img/bin/bc_geodilat.pgm
sub img/bin/bc_ws_inv.pgm img/bin/bc_geodilat.pgm img/bin/bc_sub_inv.pgm
inverse img/bin/bc_sub_inv.pgm img/result/pb2q2.pgm

#Display reesult
feh img/result/pb2q2.pgm
