. ~/.pink.start

# Segment all cell
seuil img/src/bloodcells.pgm 140 img/bin/bc.pgm
inverse img/bin/bc.pgm img/bin/bc_inv.pgm

areaclosing img/bin/bc_inv.pgm 8 200 img/bin/bc_ac.pgm
inverse img/bin/bc_ac.pgm img/bin/bc_ac_inv.pgm
pgm2GA img/bin/bc_ac_inv.pgm 1 img/bin/bc.ga
GA2khalimsky img/bin/bc.ga 0 img/bin/pb2q1.pgm

pgm2GA img/src/bloodcells.pgm 0 img/bin/bloodcells.ga
GA2khalimsky img/bin/bloodcells.ga 0 img/bin/bc_kha.pgm

# Remove cell touching the edge
frame img/bin/bc_kha.pgm 2 img/bin/bc_frame.pgm
inverse img/bin/pb2q1.pgm img/bin/bc_ws_inv.pgm
geodilat img/bin/bc_frame.pgm img/bin/bc_ws_inv.pgm 4 -1 img/bin/bc_geodilat.pgm
sub img/bin/bc_ws_inv.pgm img/bin/bc_geodilat.pgm img/bin/bc_sub_inv.pgm
inverse img/bin/bc_sub_inv.pgm img/bin/pb2q2.pgm

# Distance map + watershed
dist img/bin/pb2q2.pgm 1 img/bin/bc_dmap_long.pgm
long2byte img/bin/bc_dmap_long.pgm img/bin/bc_dmap.pgm
inverse img/bin/bc_dmap.pgm img/bin/bc_dmap_inv.pgm
pgm2GA img/bin/bc_dmap_inv.pgm 2 img/bin/bc_dmap_inv.ga
GAwatershed img/bin/bc_dmap_inv.ga img/bin/bc_ws.ga
GA2khalimsky img/bin/bc_ws.ga 0 img/bin/bc_ws_k.pgm

# Resize img
pgm2GA img/bin/bc_ws_k.pgm 1 img/bin/bc_ws_k.ga
GA2khalimsky img/bin/bc_ws_k.ga 0 img/bin/bc_ws_sized.pgm
sub img/bin/bc_ws_sized.pgm img/bin/pb2q2.pgm img/result/pb2q3.pgm

#Statistics
nbCell=$(nbcomp img/result/pb2q3.pgm 8 min)
echo "Nombre de cellule : $nbCell"
nbPixel=$(area img/result/pb2q3.pgm)
echo "Taille moyenne d'une cellule : $(echo "scale=2; $nbPixel / $nbCell" | bc)"
