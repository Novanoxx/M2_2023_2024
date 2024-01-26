## TP1 : Parallélisme intra-processeur - OpenMP

#### Exercice 1 :

L'architecture où le TP a été réalisé compte 6 coeurs et sa fréquence d'horloge est d'environ 900 MHz.

"IMAGE A METTRE ICI"

#### Exercice 2 :

Avec le mot clé private, val2 est égale à 1
Avec le mot clé firstprivate, val2 est égale à 2001
private initie la valeur de val2 à 0 tandis que firstprivate récupère la valeur de val2 initialisé avant la boucle.

"IMAGE A METTRE ICI"

#### Exercice 3 :

"METTRE DU CODE ICI"

"IMAGE A METTRE ICI"

#### Exercice 4 :

2. Durée d'exécution monoprocesseur de référence :
t1 = 2.2874925

4. Il faut utiliser firstprivate sur x afin qu'il n'y ait pas de problème de partage de variable entre les threads, sans firstprivate, la valeur de PI sera faussée.

"METTRE DU CODE ICI"
"IMAGE A METTRE ICI"

t3 = 1.16182017

5. SpeedUp = t_mono_thread / t_x_threads
t1/t2 = 1.96888688892

On s'attendait à ce que ce soit 2 fois plus rapide, ce qui est plus ou moins le cas dans notre cas après avoir calculé l'acccélération

6.
"METTRE DU CODE ICI"
"IMAGE A METTRE ICI"

t3 = 0.79572899
t6 = 0.451937

7. On remarque qu'avec la stratégie d'ordonnancement dynamic, le programme est très lent par rapport à la façon static.
t6_DYNAMIC = 3.2063438
t6_STATIC = 0.510638

8. pas à faire

#### Exercice 5 :

1. voir code dans trash.txt

2. 