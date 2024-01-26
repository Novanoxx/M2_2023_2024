/**@file    g2x_particule.h
 * @author  Universite Gustave Eiffel
 * @author  E.Incerti - eric.incerti@univ-eiffel.fr
 * @brief   Base public control functions
 * @version 6.e
 * @date    Aug.2022 (doc generation)
 */
/**@cond SHOW_SECURITY_DEFINE */

#ifdef __cplusplus
  extern "C" {
#endif

#ifndef _G2X_PART_H
  #define _G2X_PART_H
/**@endcond                   */

  #include <g2x.h>

  /**---------------------------------------------------------------*
   * le type "particule" :
   * - une position p
   * - une vitesse dont on g�re s�par�ment
   *   - la direction u (de norme 1)
   *   - l'amplitude  d
   *---------------------------------------------------------------**/
  typedef struct
  {
    G2Xpoint  p; // position
    G2Xvector u; // direction (doit TOUJOURS etre de norme 1)
    double    d; // la norme associ�e
  }
  G2Xpart;

/**-------------------------------------------------------------------------- *
 *  constructeur 1 : un point et une vitesse (d�compos�e en direction/norme)
 *---------------------------------------------------------------------------**/
  G2Xpart   g2x_Part_pv(G2Xpoint pos, G2Xvector vit);

/**-------------------------------------------------------------------------- *
 *  constructeur 2 : un point et une direction et une norme
 *---------------------------------------------------------------------------**/
  G2Xpart   g2x_Part_pud(G2Xpoint pos, G2Xvector dir, double norm);

/**-------------------------------------------------------------------------- *
 *  constructeur 3 : un bi-point [AB] => A, direction(AB), distance(AB)
 *---------------------------------------------------------------------------**/
  G2Xpart   g2x_Part_2p(G2Xpoint A, G2Xpoint B);

/**-------------------------------------------------------------------------- *
 *  vecteur vitesse "r�el" v=d*u
 *---------------------------------------------------------------------------**/
  G2Xvector g2x_PartFullVect(G2Xpart *a);

/**-------------------------------------------------------------------------- *
 *  position de l'extremit� du vecteur p+d*u
 *---------------------------------------------------------------------------**/
  G2Xpoint  g2x_PartNextPos(G2Xpart *a);

/**-------------------------------------------------------------------------- *
 *  d�placement de la particule p <- p+d*u
 *  �quivalent � a->p = g2x_PartNextPos(a);
 *---------------------------------------------------------------------------**/
  void      g2x_PartMove(G2Xpart *a);

/**-------------------------------------------------------------------------- *
 *  traverse les bords de la fen�tre pour revenir de l'autre c�t�
 *---------------------------------------------------------------------------**/
  void      g2x_PartCross(G2Xpart *a);

/**-------------------------------------------------------------------------- *
 *  rebond sur les bords de la fen�tre (fa�on billard) : cin�tique inverse
 *---------------------------------------------------------------------------**/
  void      g2x_PartBounce(G2Xpart *a);

/**-------------------------------------------------------------------------- *
 *  rotation d'angle "rad" (donc en radians...) du vecteur directeur u
 *  utilise les matrices de transformation (cf. <g2x_transfo.h>)
 *---------------------------------------------------------------------------**/
  void      g2x_PartRotateRad(G2Xpart *a, double rad);

/**-------------------------------------------------------------------------- *
 *  rotation d'angle "deg" (donc en degr�s...) du vecteur directeur u
 *  utilise les matrices de transformation et les LUT cos/sin (cf. <g2x_geom.h>)
 *---------------------------------------------------------------------------**/
  void      g2x_PartRotateDegLUT(G2Xpart *a, double deg);

/**-------------------------------------------------------------------------- *
 *  affichage, juste pour prototypage d'algo
 *---------------------------------------------------------------------------**/
  void      g2x_PartDraw(G2Xpart *a, G2Xcolor col);

/**========================================================================= *
 *  DES FONCTIONALITES DE PLUS HAUT NIVEAU
 * ========================================================================= **/

/**-------------------------------------------------------------------------- *
 *  intersection des trajectoire et r�cup�ration du point de collision
 *---------------------------------------------------------------------------**/
  bool      g2x_PartInterPart(G2Xpart *a, G2Xpart *b, G2Xpoint *I);

/**-------------------------------------------------------------------------- *
 *  collision/d�viation de 2 particules dans un rayon 'dist'
 *  concr�tement, il faudra associer un 'rayon' � chaque particule et
 *  on utilisera dist = (ray_a+ray_b)
 *---------------------------------------------------------------------------**/
  bool      g2x_PartCollPart(G2Xpart *a, G2Xpart *b, double ray);

/**-------------------------------------------------------------------------- *
 *  Collision/d�viation Particule/Cercle
 *---------------------------------------------------------------------------**/
  bool      g2x_PartCollCircle(G2Xpart *a, G2Xpoint C, double r);

/**-------------------------------------------------------------------------- *
 *  Evitement/d�viation Particule/Cercle  (1.<coeff<25.)
 *---------------------------------------------------------------------------**/
  bool      g2x_PartBypassCircle(G2Xpart *a, G2Xpoint C, double r, double coeff);

/**-------------------------------------------------------------------------- *
 *  <a>  suit <b> : Va = (1.-alf)*Va + alf*dir(AB)*norme(Vb)
 *  si dist(a,b)<dist : pas de poursuite
 *  (usage classique : dist=(ray_a+ray_b -- cf. au dessus)
 *---------------------------------------------------------------------------**/
  bool      g2x_PartPursuit(G2Xpart *a, G2Xpart *b, double alf, double dist);

  bool      g2x_PartPosTrack(G2Xpart *a, G2Xpoint tar, double alf, double dist);
#endif

#ifdef __cplusplus
  }
#endif
/*!=============================================================!**/
