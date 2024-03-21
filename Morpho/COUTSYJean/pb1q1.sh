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
GA2khalimsky img/bin/uo.ga 1 img/bin/uo.ga.pgm

#Cleaner result
areaopening img/bin/watershed.pgm 8 1000 img/bin/result1.pgm
areaclosing img/bin/result1.pgm 8 1000 img/bin/result1.pgm

#Superposition of watershed and image
surimp img/bin/uo.ga.pgm img/bin/result1.pgm img/result/pb1q1.ppm

#Display superposition
feh img/result/pb1q1.ppm
