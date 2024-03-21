. ~/.pink.start

#Correct original image
heightminima img/src/uo.pgm 4 50 img/bin/out.pgm
areaopening img/bin/out.pgm 8 20 img/bin/out.pgm
areaclosing img/bin/out.pgm 8 20 img/bin/out.pgm

#Create watershed
pgm2GA img/bin/out.pgm 0 img/bin/out.ga
GAwatershed img/bin/out.ga ./img/bin/watershed.ga
GA2khalimsky img/bin/watershed.ga 0 img/bin/watershed.pgm

#Change size for initial image
pgm2GA img/src/uo.pgm 1 img/bin/uo.ga
GA2khalimsky img/bin/uo.ga 0 img/bin/uo.ga.pgm

#Remove cells that touch the frame
frame img/bin/uo.ga.pgm 2 img/bin/uo_frame.pgm
inverse img/bin/watershed.pgm img/bin/pb1q1_inv.pgm
geodilat img/bin/uo_frame.pgm img/bin/pb1q1_inv.pgm 4 -1 img/bin/geodilat.pgm
sub img/bin/pb1q1_inv.pgm img/bin/geodilat.pgm img/bin/uo_sub_inv.pgm
inverse img/bin/uo_sub_inv.pgm img/bin/pb1q2.pgm

#Cleaner result
areaopening img/bin/pb1q2.pgm 8 1000 img/bin/result2.pgm
areaclosing img/bin/result2.pgm 8 1000 img/bin/result2.pgm

#Superposition of good sized image and result (for display)
surimp img/bin/uo.ga.pgm img/bin/result2.pgm img/result/pb1q2.pgm

#Display
feh img/result/pb1q2.pgm

#Display resultat used for later
feh img/bin/result2.pgm
